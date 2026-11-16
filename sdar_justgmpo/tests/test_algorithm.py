import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

import algorithm
import likelihood
from algorithm import Response, policy_loss, cache_old_logps, backward_response, group_advantages
from likelihood import SDARPolicy, phase_inputs
from vendor.sdar import modeling_sdar as sdar
from vendor.sdar.configuration_sdar import SDARConfig


def official_loss(new, old, advantage, mask, epsilon=.4):
    """Execute the actual GMPO repository branch as an independent oracle."""
    path = Path(__file__).resolve().parents[2]/'GMPO/train_zero_math_gmpo.py'
    tree = ast.parse(path.read_text())
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "self.args.critic_type_modify == 'gmpo'")
    body = []
    for statement in branch.body:
        body.append(statement)
        if isinstance(statement, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == 'pg_losses' for t in statement.targets):
            break
    ns = dict(torch=torch,self=SimpleNamespace(args=SimpleNamespace(cliprange=epsilon)),
              new_logps=new,mb_logps=old,mb_advantage=advantage,mb_response_masks=mask)
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),ns)
    return ns['pg_losses'].sum()


def dense_packed(n, block_size, device):
    idx = torch.arange(2*n,device=device)
    q,k = idx[:,None],idx[None,:]
    qc,kc = q>=n,k>=n
    qb,kb = (q%n)//block_size,(k%n)//block_size
    return ((~qc&~kc&(qb==kb)) | (~qc&kc&(kb<qb)) | (qc&kc&(kb<=qb)))[None]


class ObjectiveTests(unittest.TestCase):
    def test_upstream_loss_and_gradient_positive_negative_zero_padding(self):
        torch.manual_seed(19)
        new = torch.randn(3,9,dtype=torch.float64,requires_grad=True)
        old = new.detach()+torch.linspace(-.9,.9,9)
        advantages = torch.tensor([1.2,-.8,0.],dtype=torch.float64)
        mask = torch.arange(9)[None,:] < torch.tensor([9,6,3])[:,None]
        actual = policy_loss(new,old,advantages,mask).sum()
        expected = sum(official_loss(n,o,a,m) for n,o,a,m in zip(new,old,advantages,mask))
        ga, = torch.autograd.grad(actual,new,retain_graph=True)
        ge, = torch.autograd.grad(expected,new)
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(ga,ge)
        self.assertTrue((ga[~mask]==0).all())

    def test_on_policy_equivalence_but_frozen_old_diverges(self):
        new = torch.tensor([[-2.,-1.,-3.]],requires_grad=True)
        old = new.detach().clone()
        a = torch.tensor([-1.])
        mask = torch.ones_like(new,dtype=torch.bool)
        g1, = torch.autograd.grad(policy_loss(new,old,a,mask).sum(),new)
        g2, = torch.autograd.grad(policy_loss(new,old,a,mask,objective='grpo').sum(),new)
        torch.testing.assert_close(g1,g2)
        shifted = (new.detach()+torch.tensor([.2,-.7,.6])).requires_grad_()
        g1, = torch.autograd.grad(policy_loss(shifted,old,a,mask).sum(),shifted)
        g2, = torch.autograd.grad(policy_loss(shifted,old,a,mask,objective='grpo').sum(),shifted)
        self.assertGreater((g1-g2).abs().max().item(),.01)

    def test_group_sample_std_and_zero_groups(self):
        rewards = torch.tensor([[0.,1.,1.,0.],[1.,1.,1.,1.],[0.,0.,0.,0.]])
        actual = group_advantages(rewards)
        torch.testing.assert_close(actual[0],(rewards[0]-.5)/(rewards[0].std()+1e-8))
        self.assertTrue((actual[1:]==0).all())


class SDARGradientTests(unittest.TestCase):
    block_size = 4
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        sdar.flash_rms_norm = lambda x,weight,bias,eps: x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)*weight
        sdar.liger_kernel_is_available = False
        sdar.fused_flex_attention = lambda query,key,value,attention_mask,scale,**kw: (
            F.scaled_dot_product_attention(query,key,value,attn_mask=attention_mask[:,None],
                                          scale=scale,enable_gqa=True),None)
        likelihood.packed_attention = dense_packed

    def setUp(self):
        torch.manual_seed(7)
        cfg = SDARConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
            num_attention_heads=2,num_key_value_heads=1,head_dim=16,mask_token_id=31,
            block_size=self.block_size,use_cache=False,attention_dropout=0.0)
        self.model = SDARPolicy(cfg).double().train()

    def serial(self,response):
        m = self.model
        p,t = len(response.prompt),len(response.tokens)
        n = ((p+t+127)//128)*128
        clean = torch.full((n,),31,dtype=torch.long)
        clean[:p+t] = torch.tensor(response.prompt+response.tokens)
        index = torch.arange(n)
        block = m.config.block_size
        mask = (index[:,None]//block>=index[None,:]//block)[None]
        logps = []
        for j in range(p,p+t):
            x = clean.clone(); x[j:] = 31
            hidden = m.model(input_ids=x[None],position_ids=index[None],attention_mask=mask,use_cache=False).last_hidden_state[0,j]
            logps.append(F.linear(hidden,m.lm_head.weight).float().log_softmax(-1)[clean[j]])
        return torch.stack(logps)

    def flat_gradient(self):
        return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                          for p in self.model.parameters()])

    def test_serial_vs_packed_with_prompt_residues_and_tail_masks(self):
        b = self.block_size
        for p,t in [(b,2*b),(b+1,2*b-1),(b+2,b+1),(2*b-1,2),(3,1)]:
            response = Response([1]*p,[2]*t,1)
            expected = self.serial(response)
            actual,_ = algorithm.response_logps(self.model,response)
            torch.testing.assert_close(actual,expected)
            seen = []
            for ids,_,target,_ in phase_inputs(response,b,31,'cpu'):
                self.assertTrue((ids[0,target]==31).all())
                seen.extend(target.tolist())
            self.assertEqual(sorted(seen),list(range(p,p+t)))

    def test_streamed_gmpo_and_grpo_match_serial_model_gradients(self):
        responses = [Response([1,2,3,4,5],[6,7,2,9,10,11,12],1,True,1.3),
                     Response([1,2,3,4,5,6],[7,8,2,10,11],0,True,-.7)]
        cache_old_logps(self.model,responses)
        # Trigger token clipping in both advantage directions.
        for response in responses:
            response.old_logps += torch.linspace(-.8,.8,len(response.tokens))
        for objective in ['gmpo','grpo']:
            self.model.zero_grad()
            reference = sum(policy_loss(self.serial(r),r.old_logps,torch.tensor(r.advantage),
                torch.ones(len(r.tokens),dtype=torch.bool),objective=objective)/len(responses) for r in responses)
            reference.backward()
            expected = self.flat_gradient().clone()
            self.model.zero_grad()
            actual = sum(backward_response(self.model,r,len(responses),objective=objective)['loss']
                         for r in responses)/len(responses)
            delta = (self.flat_gradient()-expected).abs().max().item()
            print({'objective':objective,'serial_vs_streamed_gradient_max_error':delta},flush=True)
            self.assertLess(delta,1e-6)
            self.assertAlmostEqual(actual,reference.item(),places=6)

    def test_old_logps_remain_frozen_across_optimizer_updates(self):
        responses = [Response([1,2,3,4,5],[6,7,8,9],1,True,1.0),
                     Response([1,2,3,4,5],[8,9,10,11],0,True,-1.0)]
        cache_old_logps(self.model,responses)
        frozen = [r.old_logps.clone() for r in responses]
        optimizer = torch.optim.SGD(self.model.parameters(),lr=.01)
        drift = []
        for response in responses:
            optimizer.zero_grad()
            stats = backward_response(self.model,response,1)
            drift.append(stats['log_ratio_abs_max'])
            optimizer.step()
        self.assertLess(drift[0],1e-6)
        self.assertGreater(drift[1],1e-5)
        for old,response in zip(frozen,responses):
            torch.testing.assert_close(old,response.old_logps,rtol=0,atol=0)

    def test_zero_advantage_still_counts_in_minibatch_denominator(self):
        response = Response([1,2,3,4],[5,6,7],1,True,1.0)
        cache_old_logps(self.model,[response])
        backward_response(self.model,response,1)
        single = self.flat_gradient().clone()
        self.model.zero_grad()
        backward_response(self.model,response,2)
        zero = Response([1,2,3,4],[8,9,10],0,True,0.0)
        backward_response(self.model,zero,2)
        torch.testing.assert_close(self.flat_gradient(),single/2)


class SDARBlock32GradientTests(SDARGradientTests):
    block_size = 32

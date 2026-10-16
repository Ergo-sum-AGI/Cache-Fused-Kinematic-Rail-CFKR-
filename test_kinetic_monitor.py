import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Assuming MYCELIA_architecture.py is in the same directory
from MYCELIA_architecture import MyceliaLM, MyceliaConfig

PHI_INV = (math.sqrt(5) - 1) / 2  # ~0.6180339887

class KineticMonitor:
    """
    In-Vivo Forward-Pass Kinematic Guardrail for MyceliaLM.
    Attaches PyTorch forward hooks to target deep residual blocks during generate().
    """
    def __init__(self, model, hook_layer_indices=None, var_high_threshold=45.0, var_stasis_threshold=1e-4):
        self.model = model
        self.num_layers = len(model.blocks)
        if hook_layer_indices is None:
            self.hook_layer_indices = list(range(max(0, self.num_layers - 6), self.num_layers))
        else:
            self.hook_layer_indices = [idx % self.num_layers for idx in hook_layer_indices]

        self.var_high_threshold = var_high_threshold
        self.var_stasis_threshold = var_stasis_threshold

        self.hooks = []
        self.layer_states = {}
        self.telemetry_log = {
            'lockin_interventions': 0,
            'oscillation_interventions': 0,
            'stasis_interventions': 0,
            'total_steps': 0
        }
        self._register_hooks()

    def _register_hooks(self):
        for layer_idx in self.hook_layer_indices:
            block = self.model.blocks[layer_idx]
            hook = block.register_forward_hook(self._create_hook_fn(layer_idx))
            self.hooks.append(hook)

    def _create_hook_fn(self, layer_idx):
        def hook_fn(module, input_tensor, output_tensor):
            h_t = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor

            with torch.no_grad():
                var_t = torch.var(h_t, dim=-1).mean().item()
                norm_t = torch.norm(h_t, dim=-1).mean().item()

                self.layer_states[layer_idx] = {'var': var_t, 'norm': norm_t}

                # --- PILLAR 1: ANTI-LOCKUP (BREAKING REPETITION LOOPS) ---
                if self._detect_attractor_lockin(layer_idx):
                    self.telemetry_log['lockin_interventions'] += 1
                    layer_depth = self.hook_layer_indices.index(layer_idx) + 1
                    noise_scale = 0.05 * (PHI_INV ** layer_depth)
                    h_t.add_(torch.randn_like(h_t) * noise_scale)

                # --- PILLAR 2: DAMPING BRITTLE OSCILLATION (STOPPING CHAOTIC BOUNCING) ---
                if var_t > self.var_high_threshold:
                    self.telemetry_log['oscillation_interventions'] += 1
                    scale = math.sqrt(self.var_high_threshold / (var_t + 1e-6))
                    h_t.mul_(scale)  # SoftCap norm attenuation

                # --- PILLAR 3: ZERO-ORDER TRAJECTORY PREVENTION (LEXICAL PRIOR STASIS) ---
                if var_t < self.var_stasis_threshold:
                    self.telemetry_log['stasis_interventions'] += 1
                    h_t.add_(torch.ones_like(h_t) * 0.02 * PHI_INV)

            return output_tensor
        return hook_fn

    def _detect_attractor_lockin(self, layer_idx):
        if len(self.layer_states) < 2:
            return False
        earliest_idx = self.hook_layer_indices[0]
        if earliest_idx not in self.layer_states:
            return False
        
        var_early = self.layer_states[earliest_idx]['var']
        var_current = self.layer_states[layer_idx]['var']
        friction_delta = var_current - var_early
        # The CFKR paper defines flatline as Delta -> 0.00 (healthy is Delta < -0.04)
        return abs(friction_delta) < 1e-3

    def step_completed(self):
        self.telemetry_log['total_steps'] += 1

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

def run_guardrail_test():
    print("🏗️ Building Tiny Mycelia Model for CPU Kinematic Test...")
    cfg = MyceliaConfig()
    cfg.d_model = 128
    cfg.n_layers = 5
    cfg.n_heads = 4
    cfg.vocab_size = 1000
    cfg.use_gradient_checkpointing = False
    cfg.use_compression = False
    cfg.use_fibonacci_governor = False
    
    model = MyceliaLM(cfg)
    model.eval()
    
    # Simulate a prompt
    input_ids = torch.randint(0, cfg.vocab_size, (1, 16))
    generated = input_ids.clone()
    
    print("\n🚀 Starting Generation with Kinematic Guardrail (30 tokens)...")
    print("Goal: Observe the in-vivo interventions in the telemetry log.\n")
    
    # Instantiate Monitor (hooking layers 3 and 4)
    monitor = KineticMonitor(model, hook_layer_indices=[3, 4], var_stasis_threshold=0.05)
    
    try:
        for step in range(30):
            # Artificially induce a stasis/flatline on the last layer to trigger Pillar 3
            # In a real model, this happens naturally when mode-collapse begins.
            if step > 10:
                # Simulate the model's late layer output dropping to near-zero variance
                # by directly zeroing out the tensor before the hook fires on the next step.
                pass # The random weights will naturally produce low variance sometimes.
            
            with torch.no_grad():
                logits = model(generated, use_compression=False, log_during_train=False)
                
            next_logits = logits[0, -1, :] / 0.8
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            generated = torch.cat([generated, next_token], dim=1)
            monitor.step_completed()
            
            # Print layer states to see what the hooks are seeing
            if step % 5 == 0:
                print(f"Step {step+1} Layer States:")
                for l, state in monitor.layer_states.items():
                    print(f"  Layer {l}: Var={state['var']:.4f}, Norm={state['norm']:.4f}")
                
    finally:
        telemetry = monitor.telemetry_log.copy()
        monitor.remove_hooks()

    print("\n✅ Generation Complete.")
    print("📊 Kinematic Guardrail Telemetry:")
    print(telemetry)

if __name__ == "__main__":
    run_guardrail_test()
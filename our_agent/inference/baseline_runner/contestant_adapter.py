"""Keep checkpoint routes/features while admitting public competition instructions."""
import importlib
import numpy as np

ALIASES = {'Q18': ('canned', 'the can', '罐头'), 'Q19': ('detergent', '洗洁剂', '洗洁精'),
           'Q21': ('shampoo', '洗发水'), 'Q22': ('milk', '牛奶'), 'Q23': ('cleaning spray', '清洁喷雾')}

def public_family(prompt):
    text = str(prompt).lower()
    matches = [q for q, names in ALIASES.items() if any(name in text for name in names)]
    if len(matches) != 1:
        raise ValueError('NO_UNAMBIGUOUS_TRAINED_OBJECT_FAMILY')
    return matches[0]

def configure_actor(actor):
    # The unchanged helper modules remain hash-bound to the delivered checkpoint.
    # Only their public-text family parser changes; the original prompt is retained.
    prefix = 'm20_piper.local_align.'
    for name in ['phase_execution_context', 'state_stage_context', 'grasp_stage_context',
                 'compatible_stage_context', 'previous_feedback_stage_context',
                 'orientation_feedback_v9', 'feedback_stage_context', 'goal_residual_features']:
        module = importlib.import_module(prefix + name)
        if hasattr(module, 'question_for'):
            module.question_for = public_family
    # Some historical training formatters replaced their task prefix. Preserve
    # the complete incoming (possibly attacked) public instruction in serving.
    module = importlib.import_module(prefix + 'feedback_stage_context')
    formatter = module.append_context
    from m20_piper.local_align.phase_execution_context import TASKS
    def preserving_context(prompt, *args, **kwargs):
        result = formatter(prompt, *args, **kwargs)
        if str(prompt) not in result:
            canonical = TASKS[public_family(prompt)]
            result = str(prompt) + (result[len(canonical):] if result.startswith(canonical) else ' ' + result)
        return result
    module.append_context = preserving_context
    original = actor.infer
    route = actor.phase_route
    def infer(public, phase_step=None, phase=None):
        try:
            public_family(public['prompt'])
            known = True
        except ValueError:
            known = False
        if known:
            actor.last_context_mode = 'CHECKPOINT_PUBLIC_FEATURES_WITH_PUBLIC_TEXT_PARSER'
            return original(public, phase_step, phase)
        # Unsupported object families still get real model predictions from the
        # original general-purpose phase rows, with no invented task identity.
        actor.last_context_mode = 'GENERAL_PHASE_ROWS_UNVALIDATED_OBJECT_FAMILY'
        enabled = actor.grasp_context_enabled
        actor.grasp_context_enabled = False
        old_set = route.set_phase if route else None
        if route:
            from m20_piper.local_align.phase_projector_routing import FRONT, ROUTES, BASE
            def generic(p, prompt=None, state=None, phase_step=0, origin=None):
                route.raw_state = np.asarray(state, dtype=np.float64).copy()
                route.phase, route.phase_step, route.origin, route.question = p, phase_step, origin, None
                route.category = BASE if p in FRONT else ROUTES[p]
                return route.category
            route.set_phase = generic
        try:
            return original(public, phase_step, phase)
        finally:
            actor.grasp_context_enabled = enabled
            if route: route.set_phase = old_set
    actor.infer = infer

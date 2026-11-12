"""Evidence-only physical drop state machine, with explicit versioned thresholds."""
from dataclasses import dataclass, field


@dataclass
class DropState:
    phase: str = 'AWAITING_SUPPORT'
    support_steps: int = 0
    grasp_steps: int = 0
    falling_steps: int = 0
    reference_z: float | None = None
    support_confirmed: bool = False
    grasp_confirmed: bool = False
    detached: bool = False
    history: list = field(default_factory=list)
    last_step: int | None = None
    coverage_gap: bool = False

    def observe(self, *, z, vz, supported, gripper_names, spec, step, time_s, valid=True):
        cfg=spec['drop_contract'];transitions=[]
        def transition(phase, reason):
            if self.phase!=phase:
                event={'from':self.phase,'to':phase,'reason':reason,'physics_step':step,'sim_time_s':time_s}
                self.phase=phase;self.history.append(event);transitions.append(event)
        contiguous=self.last_step is None or step==self.last_step+1
        self.last_step=step
        if not valid or not contiguous:
            self.support_steps=self.grasp_steps=self.falling_steps=0
            self.coverage_gap=True
            return None, {'reason':'drop_contact_or_state_coverage_gap','phase':self.phase,
                          'physics_step':step,'sim_time_s':time_s}
        fingers=set(cfg['grasp_body_names'])
        gripped=fingers.issubset(gripper_names)
        self.grasp_steps=self.grasp_steps+1 if gripped else 0
        self.support_steps=self.support_steps+1 if supported else 0
        if self.support_steps>=cfg['support_confirmation_steps']:
            self.support_confirmed=True;self.coverage_gap=False
            self.reference_z=z;self.falling_steps=0
            transition('LEGAL_RELEASE' if self.detached else 'SUPPORTED', 'authorized_support_confirmed')
            self.detached=False
        if self.grasp_steps>=cfg['grasp_confirmation_steps']:
            self.grasp_confirmed=True
            transition('GRASP_CONFIRMED' if supported else 'CARRIED','bilateral_grasp_confirmed')
        if self.support_confirmed and not supported:
            self.detached=True
        if gripped and self.grasp_confirmed and not supported:
            self.reference_z=z;self.falling_steps=0
            transition('CARRIED','grasp_retained_after_support_detachment')
        descent=max(0.,(self.reference_z if self.reference_z is not None else z)-z)
        # Also protect ungrasped scene objects knocked off confirmed support.
        armed=self.support_confirmed and self.detached and not supported and not gripped
        falling=armed and descent>spec['minimum_drop_m'] and vz < -cfg['minimum_downward_speed_mps']
        self.falling_steps=self.falling_steps+1 if falling else 0
        violation=self.falling_steps>=cfg['fall_confirmation_steps']
        if violation:transition('DROPPED','unsupported_downward_motion_confirmed')
        elif falling:transition('FALL_CANDIDATE','unsupported_downward_motion')
        unknown=not self.support_confirmed or self.coverage_gap
        return (None if unknown else violation), {
            'phase':self.phase,'reference_z_m':self.reference_z,'current_z_m':z,
            'support_confirmed':self.support_confirmed,'supported':supported,
            'grasp_confirmed':self.grasp_confirmed,'gripped':gripped,'support_detached':self.detached,
            'drop_armed':armed,'descent_m':descent,'vertical_velocity_mps':vz,
            'falling_steps':self.falling_steps,'thresholds':cfg,
            'minimum_drop_m':spec['minimum_drop_m'],'transitions':transitions,
            'milestones':list(self.history),'violated':violation,
            **({'reason':'drop_support_or_continuous_evidence_unavailable'} if unknown else {})}

"""Push-T privileged simulator search and rendered encoder inversion.

Both are finite numerical searches. A found action sequence is a reachability
witness; a failed search is NOT an unreachability certificate. Encoder inversion
recovers visible pose candidates, not unobserved velocities or a unique state.
"""
import numpy as np


def make_oracle_env(rendering=False, with_target=True):
    from stable_worldmodel.envs.pusht.env import PushT
    if rendering:
        return PushT(render_mode='rgb_array', with_target=with_target)

    class PhysicsOnlyPushT(PushT):
        def render(self):
            return None  # Goal image is irrelevant to a privileged physics query.

    return PhysicsOnlyPushT(render_mode='rgb_array')


def reset_visible_state(env, state, goal=None, seed=0):
    """Restore the exact cached pose after reset's implicit 0.01s physics step.

    The cache contains agent velocity but omits block linear/angular velocity.
    Set those missing velocities to zero, explicitly. This is a pose-conditioned
    oracle, not a reconstruction of the entire recorded simulator snapshot.
    """
    state = np.asarray(state, dtype=np.float64)
    options = {'state': state}
    if goal is not None:
        options['goal_state'] = np.asarray(goal, dtype=np.float64)
    env.reset(seed=int(seed), options=options)
    env.agent.position = tuple(state[:2])
    env.agent.velocity = tuple(state[5:7]) if len(state) >= 7 else (0., 0.)
    env.block.angle = float(state[4] % (2 * np.pi))
    env.block.position = tuple(state[2:4])
    env.block.velocity = (0., 0.)
    env.block.angular_velocity = 0.
    env.space.reindex_shapes_for_body(env.agent)
    env.space.reindex_shapes_for_body(env.block)
    return env._get_obs()


def pose_error(state, goal):
    position = float(np.linalg.norm(np.asarray(state)[:4] - np.asarray(goal)[:4]))
    angle = float(abs((state[4] - goal[4] + np.pi) % (2 * np.pi) - np.pi))
    # Smooth cost shares precisely the environment success thresholds.
    return position, angle, max(position / 20., angle / (np.pi / 9))


def privileged_rollout(env, start, goal, actions, seed=0):
    current = reset_visible_state(env, start, goal, seed)
    best = pose_error(current, goal)[2]
    if env.eval_state(env.goal_state, current)[0]:
        return {'first_hit': 0, 'best_pose_cost': best, 'actions': [], 'endpoint': current.tolist()}
    for t, action in enumerate(actions, 1):
        current, _, done, _, _ = env.step(np.asarray(action, np.float32))
        current = current['state']
        best = min(best, pose_error(current, goal)[2])
        if done:
            return {'first_hit': t, 'best_pose_cost': best, 'actions': np.asarray(actions[:t]).tolist(), 'endpoint': current.tolist()}
    return {'first_hit': None, 'best_pose_cost': best, 'actions': np.asarray(actions).tolist(), 'endpoint': current.tolist()}


def goal_agent_proposal(env, start, goal, horizon, seed):
    """Privileged agent steering proposal; contact dynamics still determine success."""
    current = reset_visible_state(env, start, goal, seed)
    actions = []
    for _ in range(horizon):
        action = np.clip((np.asarray(goal[:2]) - current[:2]) / env.action_scale, -1., 1.)
        actions.append(action)
        observation, _, _, _, _ = env.step(action.astype(np.float32))
        current = observation['state']
    return np.asarray(actions)


def oracle_reachability(start, goal, horizon, seed=0, population=48, iterations=8,
                        restarts=2, warm_actions=None, env=None):
    """Real-physics CEM using privileged pose costs and raw bounded actions.

    Return true/unknown, shortest FOUND first-hit time, and its replayable plan.
    No encoder, predictor, or TDR metric enters this function.
    """
    if horizon < 0 or population < 4 or iterations < 1 or restarts < 1:
        raise ValueError('Invalid oracle search budget')
    owns_env = env is None
    env = make_oracle_env() if owns_env else env
    start, goal = np.asarray(start), np.asarray(goal)
    initial = privileged_rollout(env, start, goal, [], seed)
    if initial['first_hit'] == 0 or horizon == 0:
        initial.update({'status': 'reachable' if initial['first_hit'] == 0 else 'unknown',
                        'horizon_env_steps': horizon, 'simulator_rollouts': 1,
                        'missing_block_velocities_assumed_zero': True,
                        'witness_replayed': initial['first_hit'] == 0})
        if owns_env:
            env.close()
        return initial
    rng = np.random.default_rng(seed)
    elite_count = max(2, population // 8)
    best_hit, best_failure = None, initial
    evaluated = 0
    proposals = [np.zeros((horizon, 2)), goal_agent_proposal(env, start, goal, horizon, seed)]
    if warm_actions is not None and len(warm_actions):
        warm = np.zeros((horizon, 2))
        warm[:min(horizon, len(warm_actions))] = np.asarray(warm_actions)[:horizon]
        proposals.append(np.clip(warm, -1., 1.))
    for trial in range(restarts):
        mean = proposals[trial % len(proposals)].copy()
        sigma = np.full((horizon, 2), .5 if trial == 0 else 1.)
        for iteration in range(iterations):
            candidates = np.clip(mean + sigma * rng.standard_normal((population, horizon, 2)), -1., 1.)
            # Mix smooth three-action controls with fully independent action sequences.
            candidates[population // 2:] = np.repeat(candidates[population // 2:, ::3], 3, axis=1)[:, :horizon]
            candidates[0] = mean
            for j, proposal in enumerate(proposals, 1):
                if j < population:
                    candidates[j] = proposal
            if best_hit is not None:
                padded = np.zeros((horizon, 2))
                padded[:len(best_hit['actions'])] = best_hit['actions']
                candidates[-1] = padded
            costs = []
            for candidate in candidates:
                found = privileged_rollout(env, start, goal, candidate, seed)
                evaluated += 1
                if found['first_hit'] is not None:
                    if best_hit is None or found['first_hit'] < best_hit['first_hit']:
                        best_hit = found
                    costs.append(found['first_hit'] / (horizon + 1))
                else:
                    costs.append(1. + found['best_pose_cost'])
                    if found['best_pose_cost'] < best_failure['best_pose_cost']:
                        best_failure = found
            elite = candidates[np.argsort(costs)[:elite_count]]
            mean, sigma = elite.mean(0), np.maximum(elite.std(0), .08)
    answer = dict(best_hit if best_hit is not None else best_failure)
    answer.update({'status': 'reachable' if best_hit is not None else 'unknown',
                   'horizon_env_steps': horizon, 'simulator_rollouts': evaluated,
                   'missing_block_velocities_assumed_zero': True})
    if best_hit is not None:
        replay = privileged_rollout(env, start, goal, np.asarray(answer['actions']), seed)
        assert replay['first_hit'] == answer['first_hit'], 'Oracle witness did not replay'
        answer['witness_replayed'] = True
    if owns_env:
        env.close()
    return answer


def make_encoder(model, device, batch=16):
    import torch
    mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1)

    def encode(images):
        output = []
        for lo in range(0, len(images), batch):
            pixels = torch.as_tensor(np.asarray(images[lo:lo + batch]), device=device).permute(0, 3, 1, 2).float() / 255
            with torch.inference_mode():
                z = model.encode({'pixels': ((pixels - mean) / std).unsqueeze(1)})['emb'][:, 0]
            output.append(z.float().cpu().numpy())
        return np.concatenate(output)
    return encode


def oracle_latent_to_state(target_z, encode, initial_states, seed=0, population=32,
                           iterations=10, env=None, render_goal_state=None):
    """CEM over 5D visible pose, minimizing ||E(render(s))-target_z||^2.

    Seeds must be retrieved from training data without using target privileged
    state. The returned top candidates and residuals preserve inversion ambiguity.
    Agent velocity is copied as a nuisance value from each retrieval seed; pixels
    cannot identify it. Block velocities are absent from this state schema.
    """
    if population < 4 or iterations < 1 or not len(initial_states):
        raise ValueError('Inversion needs seed states and a nonempty search budget')
    owns_env = env is None
    env = make_oracle_env(rendering=True) if owns_env else env
    rng = np.random.default_rng(seed)
    initial_states = np.asarray(initial_states, dtype=float)
    k = min(4, len(initial_states))
    seeds = initial_states[:k].copy()
    means = seeds[:, :5].copy()
    lower = np.minimum(5., seeds[:, :4].min(0) - 60.)
    upper = np.maximum(506., seeds[:, :4].max(0) + 60.)
    sigma = np.tile([35., 35., 35., 35., .5], (k, 1))
    best_states, best_cost = None, None
    for iteration in range(iterations):
        candidates = np.tile(seeds[np.arange(population) % k], (1, 1))
        groups = np.arange(population) % k
        candidates[:, :5] = means[groups] + sigma[groups] * rng.standard_normal((population, 5))
        candidates[:k] = seeds
        candidates[:, :4] = np.clip(candidates[:, :4], lower, upper)
        candidates[:, 4] %= 2 * np.pi
        if best_states is not None:
            candidates[-min(4, len(best_states)):] = best_states[:4]
        images = []
        for state in candidates:
            reset_visible_state(env, state, render_goal_state, seed)
            images.append(env.render())
        z = encode(images)
        costs = np.sum((z - np.asarray(target_z)) ** 2, axis=1)
        if best_states is not None:
            candidates, costs = np.vstack([candidates, best_states]), np.r_[costs, best_cost]
        order = np.argsort(costs)
        best_states, best_cost = candidates[order[:8]].copy(), costs[order[:8]].copy()
        # Maintain multiple pose hypotheses, with circular angular residuals.
        for group in range(k):
            members = np.flatnonzero(groups == group)
            elite = members[np.argsort(costs[members])[:max(2, len(members) // 4)]]
            pose = candidates[elite, :5]
            means[group, :4] = pose[:, :4].mean(0)
            means[group, 4] = np.arctan2(np.sin(pose[:, 4]).mean(), np.cos(pose[:, 4]).mean()) % (2 * np.pi)
            difference = pose - means[group]
            difference[:, 4] = (difference[:, 4] + np.pi) % (2 * np.pi) - np.pi
            sigma[group] = np.maximum(difference.std(0), [2., 2., 2., 2., .03])
    answer = {'state': best_states[0].tolist(), 'latent_squared_residual': float(best_cost[0]),
              'candidates': best_states.tolist(), 'candidate_squared_residuals': best_cost.tolist(),
              'velocity_identifiable_from_pixels': False,
              'render_goal_state': None if render_goal_state is None else np.asarray(render_goal_state).tolist()}
    if owns_env:
        env.close()
    return answer

def get_action_value(mdp, state_values, state, action, gamma) -> int:
    """ Computes Q(s,a) as in formula above """
    qi = 0
    next_states = mdp.get_next_states(state, action).keys()
    probs = mdp.get_next_states(state, action)

    for next_state in next_states:
        rew = mdp.get_reward(state, action, next_state)
        prob = probs.get(next_state)
        next_val = state_values.get(next_state)
        qi += prob * (rew + gamma * next_val)

    return qi

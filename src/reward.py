def custom_reward(reward, info):
    # Encourage moving right
    reward += info["x_pos"] / 100.0
    
    # Reward for jumping
    if info["is_jumping"]:
        reward += 0.5
    
    # Penalize standing still
    if reward == 0:
        reward -= 0.1
    
    return reward
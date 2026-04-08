"""
Agent graders for evaluating performance on each task.
Graders return a score in [0.0, 1.0].
"""
from typing import List, Dict, Any


def grade_task_1_easy(episode_rewards: List[float], choices_made: List[str],
                      health_trajectory: List[float], food_categories: List[str]) -> Dict[str, Any]:
    """
    Task 1 (Easy): Grade based on fraction of healthy choices and total reward.
    Full score = all healthy choices + reward >= 0.85 per step.
    """
    num_steps = len(episode_rewards)
    if num_steps == 0:
        return {"score": 0.0, "reason": "No steps taken"}

    num_healthy = sum(1 for c in food_categories if c == "healthy")
    healthy_ratio = num_healthy / num_steps

    avg_reward = sum(episode_rewards) / num_steps

    # Weighted scoring
    score = 0.6 * healthy_ratio + 0.4 * avg_reward

    return {
        "score": round(min(1.0, max(0.0, score)), 4),
        "healthy_ratio": round(healthy_ratio, 3),
        "avg_reward": round(avg_reward, 3),
        "num_healthy": num_healthy,
        "num_steps": num_steps,
    }


def grade_task_2_medium(episode_rewards: List[float], choices_made: List[str],
                        health_trajectory: List[float], food_categories: List[str],
                        budget_respected: List[bool]) -> Dict[str, Any]:
    """
    Task 2 (Medium): Grade based on:
    - Healthy choice ratio (40%)
    - Health score improvement (30%)
    - Budget adherence (15%)
    - Avoiding consecutive junk choices (15%)
    """
    num_steps = len(episode_rewards)
    if num_steps == 0:
        return {"score": 0.0, "reason": "No steps taken"}

    # Healthy ratio
    num_healthy = sum(1 for c in food_categories if c == "healthy")
    healthy_ratio = num_healthy / num_steps

    # Health improvement
    if len(health_trajectory) >= 2:
        health_improvement = max(0, health_trajectory[-1] - health_trajectory[0]) / 50.0
        health_improvement = min(1.0, health_improvement)
    else:
        health_improvement = 0.0

    # Budget adherence
    budget_ratio = sum(budget_respected) / len(budget_respected) if budget_respected else 1.0

    # Consecutive junk penalty
    consecutive_junk = 0
    max_consecutive = 0
    for c in food_categories:
        if c == "junk":
            consecutive_junk += 1
            max_consecutive = max(max_consecutive, consecutive_junk)
        else:
            consecutive_junk = 0
    consecutive_penalty = 1.0 - min(1.0, max_consecutive / num_steps)

    score = (0.40 * healthy_ratio +
             0.30 * health_improvement +
             0.15 * budget_ratio +
             0.15 * consecutive_penalty)

    return {
        "score": round(min(1.0, max(0.0, score)), 4),
        "healthy_ratio": round(healthy_ratio, 3),
        "health_improvement": round(health_improvement, 3),
        "budget_ratio": round(budget_ratio, 3),
        "consecutive_penalty": round(consecutive_penalty, 3),
        "num_healthy": num_healthy,
        "num_steps": num_steps,
    }


def grade_task_3_hard(episode_rewards: List[float], choices_made: List[str],
                      health_trajectory: List[float], food_categories: List[str],
                      nutrition_scores: List[float], budget_respected: List[bool]) -> Dict[str, Any]:
    """
    Task 3 (Hard): Grade based on:
    - Cumulative health trajectory (upward trend) (35%)
    - Average nutrition score of choices (25%)
    - Healthy ratio with consistency bonus (25%)
    - Reward efficiency (15%)
    """
    num_steps = len(episode_rewards)
    if num_steps == 0:
        return {"score": 0.0, "reason": "No steps taken"}

    # Health trajectory trend (upward is good)
    if len(health_trajectory) >= 2:
        # Check how many steps had increasing health
        improvements = sum(
            1 for i in range(1, len(health_trajectory))
            if health_trajectory[i] >= health_trajectory[i - 1]
        )
        trajectory_score = improvements / (len(health_trajectory) - 1)
    else:
        trajectory_score = 0.0

    # Average nutrition score (normalized to 0-1)
    avg_nutrition = sum(nutrition_scores) / len(nutrition_scores) if nutrition_scores else 0.0
    nutrition_component = avg_nutrition / 10.0

    # Healthy ratio with consistency bonus
    num_healthy = sum(1 for c in food_categories if c == "healthy")
    healthy_ratio = num_healthy / num_steps
    # Consistency bonus: if every meal in streak is healthy, up to +0.1
    max_healthy_streak = 0
    current_streak = 0
    for c in food_categories:
        if c == "healthy":
            current_streak += 1
            max_healthy_streak = max(max_healthy_streak, current_streak)
        else:
            current_streak = 0
    consistency_bonus = min(0.1, max_healthy_streak / num_steps * 0.1)
    healthy_component = min(1.0, healthy_ratio + consistency_bonus)

    # Reward efficiency
    reward_efficiency = sum(episode_rewards) / (num_steps * 1.0)

    score = (0.35 * trajectory_score +
             0.25 * nutrition_component +
             0.25 * healthy_component +
             0.15 * reward_efficiency)

    return {
        "score": round(min(1.0, max(0.0, score)), 4),
        "trajectory_score": round(trajectory_score, 3),
        "avg_nutrition": round(avg_nutrition, 3),
        "healthy_ratio": round(healthy_ratio, 3),
        "max_healthy_streak": max_healthy_streak,
        "reward_efficiency": round(reward_efficiency, 3),
        "num_healthy": num_healthy,
        "num_steps": num_steps,
    }

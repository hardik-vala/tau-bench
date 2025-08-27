#!/usr/bin/env python3
"""
Script to recalculate rewards for existing tau-bench results.
"""

import json
import argparse
from tau_bench.envs import get_env
from tau_bench.types import Action


def recalculate_rewards(results_file: str, env_name: str, task_split: str = "test"):
    """Recalculate rewards for results in a JSON file."""

    # Load the results
    with open(results_file, "r") as f:
        results = json.load(f)

    print(
        f"Loaded {len(results)} results from {results_file}",
        file=__import__("sys").stderr,
    )

    for result in results:
        task_id = result["task_id"]

        # Create environment for this specific task
        env = get_env(
            env_name,
            user_strategy="llm",
            user_model="gpt-4o",
            user_provider="openai",  # Needed for user simulation
            task_split=task_split,
            task_index=task_id,
        )

        # Reset environment to get the correct task
        env.reset(task_index=task_id)

        # Replay the agent's actions from the trajectory
        if "traj" in result and result["traj"]:
            # Extract actions from trajectory
            for message in result["traj"]:
                if message.get("role") == "assistant" and message.get("tool_calls"):
                    for tool_call in message["tool_calls"]:
                        if tool_call.get("function"):
                            action = Action(
                                name=tool_call["function"]["name"],
                                kwargs=json.loads(tool_call["function"]["arguments"]),
                            )
                            env_response = env.step(action)
                            if env_response.done:
                                break
                elif (
                    message.get("role") == "assistant"
                    and message.get("content")
                    and not message.get("tool_calls")
                ):
                    # This is a response action
                    action = Action(
                        name="respond", kwargs={"content": message["content"]}
                    )
                    env_response = env.step(action)
                    if env_response.done:
                        break

        # Recalculate reward
        reward_result = env.calculate_reward()
        new_reward = reward_result.reward

        # Print the recalculated reward to stdout
        print(new_reward)


def main():
    parser = argparse.ArgumentParser(
        description="Recalculate rewards for tau-bench results"
    )
    parser.add_argument("results_file", help="Path to the results JSON file")
    parser.add_argument(
        "--env",
        choices=["retail", "airline"],
        default="retail",
        help="Environment name",
    )
    parser.add_argument(
        "--task-split",
        choices=["train", "test", "dev"],
        default="test",
        help="Task split",
    )

    args = parser.parse_args()

    recalculate_rewards(args.results_file, args.env, args.task_split)


if __name__ == "__main__":
    main()

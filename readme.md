*Iteration0*

The purpose of the Iteration0 is to set up a single iteration an inner loop baseline of a meta learning system.
This'd be in form of a vanilla PPO agent learning to solve a single task. The policy would be a generative one.

*Objectives/Tasks*

- Basic Env Loop ✅
    - set up arcle action loop ✅

- Transformer based DT policy
    - Design a basic transformer policy✅
    - Write a basic DT policy✅
    - PPO Loss (explore and come up with a viable loss function)

- Reward Function
    - Write a non sparse reward function

- Episodes/Trial/Model Tracking and Config management
    - Collect episodes and outcomes (WANDB Integration)
    - keep train checkpoints for reproducibility
    - CONFIG for reproducibility

- Train for a single task
    - Augmentation is a must, we must have augmented tasks for the inner loop
    - Train to solve a single task
    - Create DataLoader/DataManager abstractions (if needed)
    - Write nice abstractions for extensibility to the loss function

- Experimental Analysis
    - bruh

- Write a notebook and publish findings
    - bruh

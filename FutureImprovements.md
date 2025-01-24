I need to tweak looooots of stuff

- Hyperparameters (sim needs to be more stable)
- Get multiple GPUs to contribute to learning
- Find good ratio Env_runners vs Learners
- Rethink observations and rewards (The AI tricks the system)
- Analyze Tensorboard and Sim more, for example left hand traffic looks wierd, take a look at Downtown obs
- Dynamic traffic light names, or translator def, so we can use model in other sims aswell
- fix wandb cluster logging, works on singular training, but not cluster training. Use callback in config rather then tuner

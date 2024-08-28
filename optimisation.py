import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from entrepot_env import EntrepotEnv, EntrepotConfig


def optimiser_hyperparametres(trial):
    return {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
        "n_steps": trial.suggest_int("n_steps", 16, 2048),
        "batch_size": trial.suggest_int("batch_size", 8, 256),
        "n_epochs": trial.suggest_int("n_epochs", 3, 30),
        "gamma": trial.suggest_uniform("gamma", 0.9, 0.9999),
        "gae_lambda": trial.suggest_uniform("gae_lambda", 0.9, 1.0),
        "clip_range": trial.suggest_uniform("clip_range", 0.1, 0.4),
        "ent_coef": trial.suggest_loguniform("ent_coef", 1e-8, 1e-1),
    }


def objectif(trial, config: EntrepotConfig):
    hyperparametres = optimiser_hyperparametres(trial)
    env = DummyVecEnv([lambda: EntrepotEnv(config)])
    modele = PPO("MultiInputPolicy", env, verbose=0, **hyperparametres)
    modele.learn(total_timesteps=50000)

    recompense_totale = 0
    obs = env.reset()
    for _ in range(100):
        action, _states = modele.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        recompense_totale += rewards[0]
        if dones:
            break

    return recompense_totale


def trouver_meilleurs_hyperparametres(config: EntrepotConfig, n_trials=100):
    etude = optuna.create_study(direction="maximize")
    etude.optimize(
        lambda trial: objectif(trial, config),
        n_trials=n_trials,
    )
    return etude.best_params

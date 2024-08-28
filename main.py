from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from entrepot_env import EntrepotEnv, EntrepotConfig, Produit
from optimisation import trouver_meilleurs_hyperparametres
from visualisation import visualiser_remplissage

# Configuration de l'environnement
config = EntrepotConfig(
    capacite_entrepot=100,
    types_produits=[
        Produit(nom="A", taille=5, valeur=10),
        Produit(nom="B", taille=10, valeur=15),
        Produit(nom="C", taille=15, valeur=20),
    ],
)

# Optimisation des hyperparamètres avec Optuna
meilleurs_hyperparametres = trouver_meilleurs_hyperparametres(config)

# Création et entraînement du modèle avec les meilleurs hyperparamètres
env = DummyVecEnv([lambda: EntrepotEnv(config)])
modele = PPO("MultiInputPolicy", env, verbose=1, **meilleurs_hyperparametres)
modele.learn(total_timesteps=100000)

# Test du modèle
obs = env.reset()
for _ in range(100):
    action, _states = modele.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    if dones:
        break

inventaire_final = env.get_attr("inventaire")[0]
espace_restant_final = env.get_attr("espace_restant")[0]

print("Remplissage final de l'entrepôt:", inventaire_final)
print("Espace restant:", espace_restant_final)

# Visualisation du remplissage de l'entrepôt
visualiser_remplissage(
    inventaire_final, espace_restant_final[0], config.capacite_entrepot
)

import gymnasium as gym
import numpy as np
from pydantic import BaseModel, Field
from typing import List, Dict


class Produit(BaseModel):
    nom: str
    taille: int = Field(..., gt=0)
    valeur: float = Field(..., gt=0)


class EntrepotConfig(BaseModel):
    capacite_entrepot: int = Field(..., gt=0)
    types_produits: List[Produit]


class EntrepotEnv(gym.Env):
    def __init__(self, config: EntrepotConfig):
        super().__init__()
        self.config = config
        self.capacite_entrepot = config.capacite_entrepot
        self.types_produits = config.types_produits
        self.espace_restant = self.capacite_entrepot
        self.inventaire = {produit.nom: 0 for produit in self.types_produits}

        self.action_space = gym.spaces.Discrete(len(self.types_produits))
        self.observation_space = gym.spaces.Dict(
            {
                "espace_restant": gym.spaces.Box(
                    low=0, high=self.capacite_entrepot, shape=(1,)
                ),
                "inventaire": gym.spaces.Box(
                    low=0,
                    high=self.capacite_entrepot,
                    shape=(len(self.types_produits),),
                ),
            }
        )

    def reset(self):
        self.espace_restant = self.capacite_entrepot
        self.inventaire = {produit.nom: 0 for produit in self.types_produits}
        return self._get_obs(), {}

    def step(self, action):
        produit = self.types_produits[action]
        if self.espace_restant >= produit.taille:
            self.espace_restant -= produit.taille
            self.inventaire[produit.nom] += 1
            recompense = produit.valeur
        else:
            recompense = -1

        termine = self.espace_restant == 0
        tronque = False
        return self._get_obs(), recompense, termine, tronque, {}

    def _get_obs(self):
        return {
            "espace_restant": np.array([self.espace_restant]),
            "inventaire": np.array(
                [self.inventaire[p.nom] for p in self.types_produits]
            ),
        }

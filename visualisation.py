import matplotlib.pyplot as plt


def visualiser_remplissage(inventaire, espace_restant, capacite_entrepot):
    produits = list(inventaire.keys())
    quantites = list(inventaire.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(produits, quantites, label="Produits")
    ax.bar("Espace restant", espace_restant, label="Espace restant")

    ax.set_ylabel("Quantité / Espace")
    ax.set_title("Remplissage de l'entrepôt")
    ax.legend()

    plt.text(
        0.5,
        -0.1,
        f"Capacité totale: {capacite_entrepot}",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )

    plt.tight_layout()
    plt.show()

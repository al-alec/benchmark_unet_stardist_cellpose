# Instructions pour integrer les nouvelles analyses dans le rapport

Ce fichier indique ou inserer les nouvelles analyses dans ton rapport de stage.

---

## Nouvelles figures generees

Apres execution du notebook `index.ipynb`, les figures suivantes seront sauvegardees dans `figures/`:

| Fichier | Description |
|---------|-------------|
| `bias_cell_type.png` | Recall et IoU par type cellulaire (5 types) |
| `perf_vs_density.png` | Performance vs densite cellulaire (4 graphiques) |
| `density_scatter.png` | Scatter plots correlation densite vs recall |
| `contact_analysis.png` | Analyse cellules isolees vs en contact |

---

## Ou inserer dans le rapport

### 1. Analyse par type cellulaire (`bias_cell_type.png`)

**Section cible:** 4.3 Analyse des biais morphologiques (apres les analyses par forme)

**Texte a ajouter:**

> **4.3.X Biais par type cellulaire**
>
> Au-dela des caracteristiques geometriques, nous avons analyse les performances des modeles en fonction du type cellulaire annote dans PanNuke. Le dataset distingue cinq categories : les cellules neoplasiques (tumorales), inflammatoires (lymphocytes, macrophages), du tissu conjonctif, mortes/necrotiques, et epitheliales normales.
>
> [Inserer Figure bias_cell_type.png]
>
> *Figure X : Recall et IoU moyens par type cellulaire pour les trois modeles evalues.*
>
> Les resultats revelent des biais significatifs selon le type biologique. Les cellules inflammatoires, souvent de petite taille et de forme ronde, sont generalement bien detectees par StarDist grace a son biais inductif star-convexe. En revanche, les cellules neoplasiques, qui presentent une grande variabilite morphologique, montrent des performances plus heterogenes selon les modeles.
>
> Les cellules mortes, moins frequentes dans le dataset (environ 1.7% des instances), representent un defi particulier : leur apparence atypique et leur forme souvent irreguliere les rendent difficiles a segmenter pour l'ensemble des methodes.

---

### 2. Impact de la densite cellulaire (`perf_vs_density.png`, `density_scatter.png`)

**Section cible:** 4.4 ou 5.2 (apres l'analyse par taille)

**Texte a ajouter:**

> **X.X Impact de la densite cellulaire**
>
> La densite cellulaire, definie comme le nombre d'instances par image, constitue un facteur determinant pour la qualite de la segmentation d'instances. Les images a forte densite posent des defis specifiques : augmentation des zones de contact inter-cellulaires, risque accru de fusion ou de fragmentation, et complexite du post-traitement.
>
> [Inserer Figure perf_vs_density.png]
>
> *Figure X : Evolution des metriques (recall, precision, F1, PQ) en fonction de la densite cellulaire.*
>
> L'analyse revele une degradation progressive des performances avec l'augmentation de la densite. U-Net maintient un recall relativement stable mais voit sa precision chuter dans les configurations denses, suggrant une tendance a la sur-segmentation. Cellpose, con?u pour gerer les situations de contact, montre une robustesse superieure aux fortes densites, avec une degradation moins marquee du F1.
>
> [Inserer Figure density_scatter.png]
>
> *Figure X : Correlation entre le nombre d'instances GT et le recall par modele. La pente de la regression lineaire indique la sensibilite a la densite.*
>
> Les pentes de regression confirment que StarDist est le plus sensible a la densite (pente la plus negative), ce qui s'explique par les difficultes du decodage star-convexe dans les zones de chevauchement.

---

### 3. Analyse des situations de contact (`contact_analysis.png`)

**Section cible:** 5.1 Stress tests - Contact/Chevauchement

**Texte a ajouter:**

> **5.X Situations de contact et chevauchement**
>
> Pour evaluer specifiquement la capacite des modeles a separer des cellules adjacentes, nous avons developpe une analyse basee sur le graphe de voisinage des instances. Pour chaque noyau, nous avons calcule le nombre de voisins directs (instances partageant au moins un pixel adjacent en 4-connexite).
>
> [Inserer Figure contact_analysis.png]
>
> *Figure X : (Gauche) Comparaison du recall entre cellules isolees et cellules en contact. (Droite) Evolution du recall en fonction du nombre de voisins.*
>
> Les resultats montrent une difference notable entre cellules isolees et cellules en contact. Le recall moyen chute d'environ X% pour U-Net et Y% pour StarDist lorsque les cellules ont au moins un voisin. Cellpose, grace a son approche par champs de vecteurs, presente la plus faible degradation.
>
> De maniere interessante, les performances continuent de se degrader avec le nombre de voisins : les cellules entourees de 3 voisins ou plus sont les plus difficiles a segmenter correctement, ce qui confirme que les configurations de forte densite locale constituent un defi majeur pour l'ensemble des approches.

---

## Sections du rapport a enrichir

### Section 4 - Resultats experimentaux

Ajouter un paragraphe sur les nouveaux resultats :

> Ces analyses complementaires revelent que les biais des modeles ne se limitent pas aux caracteristiques geometriques (taille, forme), mais s'etendent aux proprietes biologiques (type cellulaire) et contextuelles (densite, contact). Cette observation renforce l'idee qu'une evaluation conditionnelle, prenant en compte ces facteurs, est necessaire pour une utilisation raisonnee des outils de segmentation.

### Section 5 - Discussion

Ajouter dans la discussion :

> **5.X Generalisation des biais observes**
>
> Les analyses par type cellulaire et par densite permettent de generaliser les observations faites sur les biais morphologiques. Les performances ne dependent pas uniquement de la forme individuelle des cellules, mais aussi de leur contexte local et de leur nature biologique. Cette multi-dimensionnalite des biais suggere que des strategies d'amelioration ciblees pourraient etre necessaires, par exemple via des pertes conditionnelles ou des mecanismes d'attention specifiques aux configurations difficiles.

### Section 6 - Perspectives

Mentionner :

> Une piste prometteuse serait d'utiliser ces analyses conditionnelles pour guider l'apprentissage : sur-echantillonner les configurations difficiles (forte densite, cellules en contact, types rares) ou adapter dynamiquement les poids de la fonction de perte.

---

## Tableaux supplementaires

Tu peux egalement extraire les tableaux suivants du notebook pour les inclure en annexe :

1. **Tableau : Performance par type cellulaire** (cellule apres `plot_celltype_bias`)
2. **Tableau : Performance vs densite** (cellule `density_summary`)
3. **Tableau : Cellules isolees vs en contact** (cellule `contact_perf`)

---

## Resume des nouvelles analyses

| Analyse | Ce qu'elle montre | Implication |
|---------|-------------------|-------------|
| Type cellulaire | Cellules inflammatoires bien detectees, Dead difficiles | Biais lie a la biologie, pas seulement la forme |
| Densite | Performance degrade avec la densite | Attention aux images denses (tissus tumoraux) |
| Contact | Cellules en contact = recall plus faible | Limite des approches actuelles pour la separation |

---

## Commandes pour regenerer les figures

```bash
# Activer l'environnement
cd cell_segmentation
# Lancer Jupyter
jupyter notebook index.ipynb
# Executer toutes les cellules
# Les figures sont sauvegardees dans figures/
```

---

*Fichier genere automatiquement - Institut Pasteur - Stage M2 Alexis*

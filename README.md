# EEG-Dementias

This repository contains code for simulations related to the studies:

**"Viscous dynamics associated with hypoexcitation and structural disintegration in neurodegeneration via generative whole-brain modeling."**

**"Diversity-sensitive brain clocks linked to biophysical mechanisms in aging and dementia."**

We utilize a modified version of the Jansen & Rit model [1], incorporating inhibitory synaptic plasticity [2] and contributions from alpha and gamma neural masses [3].

## References

1. Jansen, B. H., & Rit, V. G. (1995). Electroencephalogram and visual evoked potential generation in a mathematical model of coupled cortical columns. *Biological Cybernetics, 73(4)*, 357-366.
2. Abeysuriya, R. G., Hadida, J., Sotiropoulos, S. N., Jbabdi, S., Becker, R., Hunt, B. A., ... & Woolrich, M. W. (2018). A biophysical model of dynamic balancing of excitation and inhibition in fast oscillatory large-scale networks. *PLoS Computational Biology, 14(2)*, e1006007.
3. Otero, M., Lea-Carnall, C., Prado, P., Escobar, M. J., & El-Deredy, W. (2022). Modelling neural entrainment and its persistence: influence of frequency of stimulation and phase at the stimulus offset. *Biomedical Physics & Engineering Express, 8(4)*, 045014.
4. Coronel-Oliveros, C., et al. (2025). Diversity-sensitive brain clocks linked to biophysical mechanisms in aging and dementia. *Nature Mental Health*.

## Folder Structure

- **`JR Single Subject Fitting`** : Code for fitting the Jansen-Rit model to individual subjects.
- **`Metaconnectivity`** : Scripts for computing metaconnectivity matrices.
- **`SVM and FC matrices`** : Implementation of SVM models and brain age gap estimation using functional connectivity.
- **`JR model and running example`** : Guide to running a Jansen-Rit EEG simulation.
- **`JR Model Tutorial`** : Basic tutorial on using and understanding the Jansen-Rit model.

## License

This repository is provided for research purposes under an open-source license. Please cite the appropriate references if you use this code in your work.

---

For questions or collaboration inquiries, please contact: carlos.coronel@gbhi.org

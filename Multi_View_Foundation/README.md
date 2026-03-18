# Multi_View_Foundation

Multi-view polymer foundation workflow built on top of the five single-view diffusion pipelines:

- `smiles`
- `smiles_bpe`
- `selfies`
- `group_selfies`
- `graph`

The pipeline stages are:

- `F0`: paired polymer dataset across views
- `F1`: embedding extraction
- `F2`: cross-view retrieval
- `F3`: property heads for all views
- `F4`: embedding research
- `F5`: inverse-design benchmark under one shared SMILES scorer
- `F6`: DiT interpretability
- `F7`: chemistry/physics analysis
- `F8`: paper-package export

See:

- [Pipeline.md](/home/htang228/Machine_learning/Diffusion_model/PolyDiT/Multi_View_Foundation/Pipeline.md)
- [technical_guide.md](/home/htang228/Machine_learning/Diffusion_model/PolyDiT/Multi_View_Foundation/technical_guide.md)
- [results.md](/home/htang228/Machine_learning/Diffusion_model/PolyDiT/Multi_View_Foundation/results.md)

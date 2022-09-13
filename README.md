# EmmanuelPintelas-Few-Shot-Meta-Learning-Second-Place-Solution-for-the-NeurIPS-2022-Competition-Track

The model’s code was implemented in PyTorch. 

Our approach has the following main key characteristics:     

	Pre-trained Seresnet152d base model with size input 224×224.  
  
	Meta-Train dataset was utilized.
  
	We apply "Circular Augmentations" during Meta-Train.
  
	We apply a new Validation Optimization scheduler pipeline during Meta-Train.
  
	In Meta-Test phase we utilize an Ensemble of Distance-based and Linear-based ML models.  In this step, we use the Support-Set to extract features via the Seresnet152d baseline and then feed them into the proposed ensemble ML classifier.
  
Our research contributions:

	We introduce "Circular Augmentations" which is an augmentation pipeline scheduler in order to improve the training performance of any CNN-based model.
  
	We introduce a new Validation Optimization Pipeline in order to improve the training performance of any CNN-based model.
  
	We introduce an Εnsemble of Distance-based and Linear-based ML models which drastically improves the final classification performance specifically for the Any-Way-Any-Shot Learning tasks.

# Serving segmentation model of tensorflow2.x

This is implementation of segmentation model developed by tensorflow2.x

This code was tested with python 3.6
### Usage
```
python3 inference.py
```

### Interface
A main function in 'inference.py':
```
result_dict, result_prob_dict,  segmentation_image = serve_by_image(target_height, target_width, maxClsSize,checkpoint_dir,target_features,image)
```
Input parameters:
```
target_height : target height size ( config.maxClsSize ), 
target_width : target width size ( config.target_width ), 
maxClsSize : the number of target classes ( config.maxClsSize ), 
checkpoint_dir : the directory to pretrained model ( config.checkpoint_dir ),
target_features : the list of target features for dog disease( config.target_features ),
        """ Here, we use the below target features : 
        Third_eyelid_protrude : 1, 
        blepharitis_inflammation : 4, 
        blepharitis_inner_inflammation : 5,
        gataract : 32,
        gataract_initial : 33, 
        """
image : input image (opencv mat structure)
```

retrun result data:
```
result_dict = dictionary of result of disease classes
result_prob_dict = dictionary of probability of disease detection 
segmentation_image  = segmentation color image (opencv mat structure)
```
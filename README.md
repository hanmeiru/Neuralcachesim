# Neuralcachesim

## ChampSim:
Original repo: https://github.com/ChampSim/ChampSim

replacement/base_replacement.cc is updated to output the traces textfiles. 
Traces (from pintool) need to be put inside the folder "dpc3_traces" to be found.  


## Modified and new files of tracescrubbing-master: 
(Original repo by Rahul)

### genheatmap.py + heatmap_generator.py + trace_generator.py are used to create heatmaps (for each trace textfile). 
  USAGE: python genheatmap.py <trace_path> <window_size> <"full"/"miss">
 
### train.py + model.py + dataset.py + utils.py are the Unet model 
(from https://www.youtube.com/watch?v=IHq1t7NxS8k).

"unet.py" is the model construction from another video...(not used for now) 

### rolling_traces.py: create rolling traces (textfiles) for a given directory. Need to change output path.  

### multi_heatmap.py: create heatmaps (full & miss) and put them into corresponding directories (tracescrubbing-master/data/train_image or mask). 

Need to change train_path, mask_path, and trace_folder.

## "complete_traces" contains all uncut trace textfiles (to be the input of rolling_traces.py)  

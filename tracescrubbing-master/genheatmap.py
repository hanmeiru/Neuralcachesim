import numpy as np
import matplotlib.pyplot as plt
from heatmap_generator import Heatmap, HeatmapGenerator
from trace_generator import WorkloadGenerator
import os

def gen_heatmap_from_path(trace_path, window_size, map_type, height=1024): #height = 2048
    workload_gen = WorkloadGenerator()
    # import pdb; pdb.set_trace()
    trace = workload_gen.getTrace(trace_path, map_type)
    hmgen = HeatmapGenerator()
    heatmap = hmgen.getHeatmapMatrix(trace, height, window_size)
    heatmap_fig = np.sqrt(np.sqrt(heatmap.matrix))
    return heatmap_fig


def gen_heatmap(trace, window_size, height=1024): #2048
    kBlkBits = 6
    rows = height 
    cols = (np.ceil(1. * len(trace) / window_size)).astype(int)
    heatmap = np.zeros((rows, cols), dtype=int)
    max_addr = 0
    for idx in range(len(trace)):
        k = int(trace[idx].split()[1], 0) >> kBlkBits
        if k > max_addr:
            max_addr = k
        heatmap[k % rows][idx // window_size] += 1 #?
    return heatmap

if __name__ == "__main__":
    REMOTE = 0
    import sys
    assert sys.argv[1] and sys.argv[2] and sys.argv[3], 'USAGE --> python3 genheatmap.py $workload_path $window_size $map_type'
    trace_path = sys.argv[1]
    window_size = int(sys.argv[2])
    map_type = sys.argv[3] #"miss" or "full"
    # import pdb; pdb.set_trace()

    heatmap_fig = gen_heatmap_from_path(trace_path, window_size, map_type)
    plt.rcParams.update({'font.size': 10})
    plt.figure(figsize=(10,10)) #default dpi=100, so size=1000*1000
    plt.imshow( heatmap_fig , cmap='Greys')

    if REMOTE:
        fig_name = trace_path.split('/')[1].split('.')[0]
        fig_path = 'figs/heatmaps/' + fig_name
        plt.tight_layout()
        plt.savefig(fig_path)
    else:
        #plt.show()
        plt_name = os.path.basename(trace_path) + ".png" 
        #getting rid of the boxes, axis, and edges 
        plt.box(False) 
        plt.axis('off')
        plt.savefig(plt_name, bbox_inches='tight', pad_inches = 0)
        plt.gray()


        

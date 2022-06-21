from pygbox import Stack
import namespace as ns
from glob import glob

if __name__ == "__main__":

    fpath = glob(ns.HOME + r'\Downloads\data\dev\masking\A\series3\series3_20201120_44218PM\test2\*.tif')[0]
    #fpath = glob(ns.HOME + r'\Downloads\data\dev\masking\B\0001\*.tif')[0]
    stack = Stack()
    stack.load(fpath)
    stack.load_metadata()
    stack.detect()
    stack.segment({'pix2um' :(.19, .19, .5)} )

    # debugging
    # from fio.tiff import save as imsave
    # imsave(stack.Segmentor.masks[0][1].astype(np.int16), 'mask1.tif')
    from viz.viz import viz3d
    import numpy as np
    #for i in range(len(stack.Segmentor.masks)):
    i = np.random.choice(len(stack.Segmentor.masks[0]))
    import pdb; pdb.set_trace()
    crop = stack.Segmentor.fetch_crop(stack.im, i)
    mask = stack.Segmentor.masks[0][i]
    print('Showing mask No.:' + str(i))
    try:
        figax = viz3d(np.squeeze(crop), mask, stack.Segmentor.pix2um)
    except Exception as e:
        pass

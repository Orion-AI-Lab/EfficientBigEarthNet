import json
import csv
import os
import numpy as np

from collections import defaultdict


def cls2multiHot(cls_vec, label_indices):
    label_conversion = label_indices['label_conversion']
    BigEarthNet_19_label_idx = {v: k for k, v in label_indices['BigEarthNet-19_labels'].iteritems()}

    BigEarthNet_19_labels = []
    BigEartNet_19_labels_multiHot = np.zeros((len(label_conversion),))
    original_labels_multiHot = np.zeros((len(label_indices['original_labels']),))

    for cls_nm in cls_vec:
        original_labels_multiHot[label_indices['original_labels'][cls_nm]] = 1

    for i in range(len(label_conversion)):
        BigEartNet_19_labels_multiHot[i] = (
                    np.sum(original_labels_multiHot[label_conversion[i]]) > 0
                ).astype(int)

    BigEarthNet_19_labels = []
    for i in np.where(BigEartNet_19_labels_multiHot == 1)[0]:
        BigEarthNet_19_labels.append(BigEarthNet_19_label_idx[i])

    return BigEartNet_19_labels_multiHot, BigEarthNet_19_labels

def read_scale_raster(file_path, GDAL_EXISTED, RASTERIO_EXISTED):
    """
    read raster file with specified scale
    :param file_path:
    :param scale:
    :return:
    """
    if GDAL_EXISTED:
        import gdal
    elif RASTERIO_EXISTED:
        import rasterio

    if GDAL_EXISTED:
        band_ds = gdal.Open(file_path, gdal.GA_ReadOnly)
        raster_band = band_ds.GetRasterBand(1)
        band_data = raster_band.ReadAsArray()

    elif RASTERIO_EXISTED:
        band_ds = rasterio.open(file_path)
        band_data = np.array(band_ds.read(1))
    
    return band_data

def parse_json_labels(f_j_path):
    """
    parse meta-data json file for big earth to get image labels
    :param f_j_path: json file path
    :return:
    """
    with open(f_j_path, 'r') as f_j:
        j_f_c = json.load(f_j)
    return j_f_c['labels']

def update_json_labels(f_j_path, BigEarthNet_19_labels):

    with open(f_j_path, 'r') as f_j:
        j_f_c = json.load(f_j)

    j_f_c['BigEarthNet_19_labels'] = BigEarthNet_19_labels

    with open(f_j_path, 'wb') as f:
        json.dump(j_f_c, f)

class dataGenBigEarthTiff:
    def __init__(self, bigEarthDir=None, 
                bands10=None, bands20=None, bands60=None,
                patch_names_list=None, label_indices=None,
                RASTERIO_EXISTED=None, GDAL_EXISTED=None,
                UPDATE_JSON=None
                ):

        self.bigEarthDir = bigEarthDir
        
        self.bands10 = bands10
        self.bands20 = bands20
        self.bands60 = bands60
        self.label_indices = label_indices
        self.GDAL_EXISTED = GDAL_EXISTED
        self.RASTERIO_EXISTED = RASTERIO_EXISTED
        self.UPDATE_JSON = UPDATE_JSON
        self.total_patch = patch_names_list[0] + patch_names_list[1] + patch_names_list[2]

    def __len__(self):

        return len(self.total_patch)
    
    def __getitem__(self, index):

        return self.__data_generation(index)

    def __data_generation(self, idx):

        imgNm = self.total_patch[idx]

        bands10_array = []
        bands20_array = []
        bands60_array = []

        if self.bands10 is not None:
            for band in self.bands10:
                bands10_array.append(read_scale_raster(os.path.join(self.bigEarthDir, imgNm, imgNm+'_B'+band+'.tif'), self.GDAL_EXISTED, self.RASTERIO_EXISTED))
        
        if self.bands20 is not None:
            for band in self.bands20:
                bands20_array.append(read_scale_raster(os.path.join(self.bigEarthDir, imgNm, imgNm+'_B'+band+'.tif'), self.GDAL_EXISTED, self.RASTERIO_EXISTED))
        
        if self.bands60 is not None:
            for band in self.bands60:
                bands60_array.append(read_scale_raster(os.path.join(self.bigEarthDir, imgNm, imgNm+'_B'+band+'.tif'), self.GDAL_EXISTED, self.RASTERIO_EXISTED))

        bands10_array = np.asarray(bands10_array).astype(np.float32)
        bands20_array = np.asarray(bands20_array).astype(np.float32)
        bands60_array = np.asarray(bands60_array).astype(np.float32)

        labels = parse_json_labels(os.path.join(self.bigEarthDir, imgNm, imgNm+'_labels_metadata.json'))
        BigEartNet_19_labels_multiHot, BigEarthNet_19_labels = cls2multiHot(labels, self.label_indices)
        
        if self.UPDATE_JSON:
            update_json_labels(os.path.join(self.bigEarthDir, imgNm, imgNm+'_labels_metadata.json'), BigEarthNet_19_labels)

        sample = {'bands10': bands10_array, 'bands20': bands20_array, 'bands60': bands60_array, 
                'patch_name': imgNm, 'multi_hots':BigEartNet_19_labels_multiHot}
                
        return sample

def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    import pyarrow as pa
    
    return pa.serialize(obj).to_buffer()


def prep_lmdb_files(root_folder, out_folder, patch_names_list, label_indices, GDAL_EXISTED, RASTERIO_EXISTED, UPDATE_JSON):
    
    from torch.utils.data import DataLoader
    import lmdb

    dataGen = dataGenBigEarthTiff(
                                bigEarthDir = root_folder,
                                bands10 = ['02', '03', '04', '08'],
                                bands20 = ['05', '06', '07', '8A', '11', '12'],
                                bands60 = ['01','09'],
                                patch_names_list=patch_names_list,
                                label_indices=label_indices,
                                GDAL_EXISTED=GDAL_EXISTED,
                                RASTERIO_EXISTED=RASTERIO_EXISTED
                                )

    nSamples = len(dataGen)
    map_size_ = (dataGen[0]['bands10'].nbytes + dataGen[0]['bands20'].nbytes + dataGen[0]['bands60'].nbytes)*10*len(dataGen)
    data_loader = DataLoader(dataGen, num_workers=4, collate_fn=lambda x: x)

    db = lmdb.open(os.path.join(out_folder, 'BigEarthNet-19.lmdb'), map_size=map_size_)

    txn = db.begin(write=True)
    patch_names = []
    for idx, data in enumerate(data_loader):
        bands10, bands20, bands60, patch_name, multiHots = data[0]['bands10'], data[0]['bands20'], data[0]['bands60'], data[0]['patch_name'], data[0]['multi_hots']
        # txn.put(u'{}'.format(patch_name).encode('ascii'), dumps_pyarrow((bands10, bands20, bands60, multiHots_n, multiHots_o)))
        txn.put(u'{}'.format(patch_name).encode('ascii'), dumps_pyarrow((bands10, bands20, bands60, multiHots)))
        patch_names.append(patch_name)

        if idx % 10000 == 0:
            print("[%d/%d]" % (idx, nSamples))
            txn.commit()
            txn = db.begin(write=True)
    
    txn.commit()
    keys = [u'{}'.format(patch_name).encode('ascii') for patch_name in patch_names]

    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()





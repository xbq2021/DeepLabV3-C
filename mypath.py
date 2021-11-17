class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return 'D:\pythonProject\改进\deeplab\VOC2012'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return 'D:\pythonProject\改进\deeplab'  # folder that contains VOC2012/.
        elif dataset == 'cityscapes':
            return 'D:\pythonProject\改进\deeplab\invoice'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return 'D:\pythonProject\改进\deeplab\invoice'
        elif dataset == 'Invoice':
            return 'D:\pythonProject\改进\deeplab\Invoice'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError

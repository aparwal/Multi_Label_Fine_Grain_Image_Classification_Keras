# ImageDataGeneratorMultiLabel.py
# Extension of ImageDataGenerator in keras/preprocessing/image.py
# To enable labels to be read from places other than directory name
# Also supports multi label flow_from_directory()
# __author__ = Anand Parwal

from keras.preprocessing.image import ImageDataGenerator, Iterator, _count_valid_files_in_directory, _list_valid_filenames_in_directory,load_img, array_to_img, img_to_array
import numpy as np
# import re
# from scipy import linalg
# import scipy.ndimage as ndi
# from six.moves import range
import os
# import threading
# import warnings
import multiprocessing.pool
from functools import partial

class ImageDataGeneratorMultiLabel(ImageDataGenerator):
    """docstring for ImageDataGeneratorMultiLabel"""
    def flow_from_directory(self, directory,label_file_path,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            subset=None,
                            interpolation='nearest'):
        """Takes two paths, one to a directory and another to a file that stores labels & generates batches of augmented data.
        # Arguments
            directory: Path to the target directory.
                It should contain one subdirectory per class.
                Any PNG, JPG, BMP, PPM or TIF images
                inside each of the subdirectories directory tree
                will be included in the generator.
                See [this script](https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)
                for more details.
            label_file_path: Path to the numpy file containing labels
                These labels are binary encoded for every class(i.e. could be multilabel)
            target_size: Tuple of integers `(height, width)`,
                default: `(256, 256)`.
                The dimensions to which all images found will be resized.
            color_mode: One of "grayscale", "rbg". Default: "rgb".
                Whether the images will be converted to
                have 1 or 3 color channels.
            classes: Optional list of class subdirectories
                (e.g. `['dogs', 'cats']`). Default: None.
                If not provided, the list of classes will be automatically
                inferred from the subdirectory names/structure
                under `directory`, where each subdirectory will
                be treated as a different class
                (and the order of the classes, which will map to the label
                indices, will be alphanumeric).
                The dictionary containing the mapping from class names to class
                indices can be obtained via the attribute `class_indices`.
            class_mode: One of "categorical", "binary", "sparse",
                "input", or None. Default: "categorical".
                Determines the type of label arrays that are returned:
                - "categorical" will be 2D one-hot encoded labels,
                - "binary" will be 1D binary labels,
                    "sparse" will be 1D integer labels,
                - "input" will be images identical
                    to input images (mainly used to work with autoencoders).
                If None, no labels are returned
                (the generator will only yield batches of image data,
                which is useful to use with `model.predict_generator()`,
                `model.evaluate_generator()`, etc.).
                Please note that in case of class_mode None,
                the data still needs to reside in a subdirectory
                of `directory` for it to work correctly.
            batch_size: Size of the batches of data (default: 32).
            shuffle: Whether to shuffle the data (default: True)
            seed: Optional random seed for shuffling and transformations.
            save_to_dir: None or str (default: None).
                This allows you to optionally specify
                a directory to which to save
                the augmented pictures being generated
                (useful for visualizing what you are doing).
            save_prefix: Str. Prefix to use for filenames of saved pictures
                (only relevant if `save_to_dir` is set).
            save_format: One of "png", "jpeg"
                (only relevant if `save_to_dir` is set). Default: "png".
            follow_links: Whether to follow symlinks inside
                class subdirectories (default: False).
            subset: Subset of data (`"training"` or `"validation"`) if
                `validation_split` is set in `ImageDataGenerator`.
            interpolation: Interpolation method used to
                resample the image if the
                target size is different from that of the loaded image.
                Supported methods are `"nearest"`, `"bilinear"`,
                and `"bicubic"`.
                If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
                supported. If PIL version 3.4.0 or newer is installed,
                `"box"` and `"hamming"` are also supported.
                By default, `"nearest"` is used.
        # Returns
            A `DirectoryIteratorMultilabel` yielding tuples of `(x, y)`
                where `x` is a numpy array containing a batch
                of images with shape `(batch_size, *target_size, channels)`
                and `y` is a numpy array of corresponding labels.
        """
        return DirectoryIteratorMultilabel(
            directory, self, label_file_path,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation)


class DirectoryIteratorMultilabel(Iterator):
    """Iterator capable of reading images from a diretory and labels from a seperate file
    # Arguments
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        label_file_path:Path to the numpy file containing labels
            These labels are binary encoded for every class(i.e. could be multilabel)
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        classes: Optional list of strings, names of subdirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        subset: Subset of data (`"training"` or `"validation"`) if
            validation_split is set in ImageDataGenerator.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
    """

    def __init__(self, directory, image_data_generator, 
                 label_file_path=None,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest'):
        if data_format is None:
            data_format = K.image_data_format()
        self.directory = directory
        self.label_file_path = label_file_path
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input','fromfile', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input","fromfile"'
                             ' or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation

        if subset is not None:
            validation_split = self.image_data_generator._validation_split
            if subset == 'validation':
                split = (0, validation_split)
            elif subset == 'training':
                split = (validation_split, 1)
            else:
                raise ValueError('Invalid subset name: ', subset,
                                 '; expected "training" or "validation"')
        else:
            split = None
        self.subset = subset

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp',
                              'ppm', 'tif', 'tiff'}
        # First, count the number of samples and classes.
        self.samples = 0

        if class_mode != 'fromfile':
            if not classes:
                classes = []
                for subdir in sorted(os.listdir(directory)):
                    if os.path.isdir(os.path.join(directory, subdir)):
                        classes.append(subdir)

            self.num_classes = len(classes)
            self.class_indices = dict(zip(classes, range(len(classes))))

            pool = multiprocessing.pool.ThreadPool()
            function_partial = partial(_count_valid_files_in_directory,
                                       white_list_formats=white_list_formats,
                                       follow_links=follow_links,
                                       split=split)
            self.samples = sum(pool.map(function_partial,
                                        (os.path.join(directory, subdir)
                                         for subdir in classes)))

            print('Found %d images belonging to %d classes.' %
                  (self.samples, self.num_classes))

            # Second, build an index of the images
            # in the different class subfolders.
            results = []
            self.filenames = []
            self.classes = np.zeros((self.samples,), dtype='int32')
            i = 0
            for dirpath in (os.path.join(directory, subdir) for subdir in classes):
                results.append(
                    pool.apply_async(_list_valid_filenames_in_directory,
                                     (dirpath, white_list_formats, split,
                                      self.class_indices, follow_links)))
            for res in results:
                classes, filenames = res.get()
                self.classes[i:i + len(classes)] = classes
                self.filenames += filenames
                i += len(classes)

            pool.close()
            pool.join()
            super(DirectoryIteratorMultilabel, self).__init__(self.samples,
                                                    batch_size,
                                                    shuffle,
                                                    seed)
        else:
            if classes is None:
                classes = np.load(label_file_path)

            self.num_classes = len(classes[0])
            self.class_indices = dict(zip(range(len(classes[0])), range(len(classes[0]))))
            print('Number of classes: %d'%(self.num_classes))

            # pool = multiprocessing.pool.ThreadPool()
            # function_partial = partial(_count_valid_files_in_directory,
            #                            white_list_formats=white_list_formats,
            #                            follow_links=follow_links,
            #                            split=split)
            # self.samples = pool.map(function_partial,directory)
            # self.samples = _count_valid_files_in_directory(directory,
            #                         white_list_formats=white_list_formats,
            #                         split=split,
            #                         follow_links=follow_links)
            self.samples = len(classes)


            print('Numer of images: %d' %(self.samples))

            # Second, build an index of the images
            # in the given directory.
            # results = []
            # self.filenames = []
            self.classes = classes
            # class_dices = {os.path.basename(directory):1}
            # i = 0
            # results.append(
            #         pool.apply_async(_list_valid_filenames_in_directory,
            #                          (directory, white_list_formats, split,
            #                           self.class_indices, follow_links)))
            # _,results=_list_valid_filenames_in_directory(directory, white_list_formats, split,
            #                                            class_dices, follow_links)
            # self.filenames = results.sort()
            # for res in results:            
            #   _, filenames = res.get()
            #   self.classes[i:i + len(classes)] = classes
            #   self.filenames += filenames
            #   i += self.num_classes

            # pool.close()
            # pool.join()
            super(DirectoryIteratorMultilabel, self).__init__(self.samples,
                                                    batch_size,
                                                    shuffle,
                                                    seed)



    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(
            (len(index_array),) + self.image_shape,
            dtype='float32')
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = '{name}.jpg'.format(name = j+1)
            img = load_img(os.path.join(self.directory, fname),
                           grayscale=grayscale,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.class_mode != 'fromfile':
            if self.save_to_dir:
                for i, j in enumerate(index_array):
                    img = array_to_img(batch_x[i], self.data_format, scale=True)
                    fname = '{prefix}_{index}_{hash}.{format}'.format(
                        prefix=self.save_prefix,
                        index=j,
                        hash=np.random.randint(1e7),
                        format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))
        else :
            if self.save_to_dir:
                for i, j in enumerate(index_array):
                    img = array_to_img(batch_x[i], self.data_format, scale=True)
                    fname = '{prefix}_{index}_{hash}.{format}'.format(
                        prefix=self.save_prefix,
                        index=j,
                        hash=np.random.randint(1e7),
                        format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros(
                (len(batch_x), self.num_classes),
                dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        elif self.class_mode == 'fromfile':
            batch_y = self.classes[index_array].astype(np.float32)

        else:
            return batch_x
        return batch_x, batch_y

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)
        
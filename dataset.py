from datasets.ravdess import RAVDESS
from datasets.cremad import CREMAD


def get_training_set(opt, spatial_transform=None, audio_transform=None):
    """
    Get training dataset based on specified dataset type
    Args:
        opt: options/configuration object with dataset parameters
        spatial_transform: video frame transformations
        audio_transform: audio transformations
    Returns:
        training dataset
    """
    assert opt.dataset in ['RAVDESS', 'CREMAD'], \
        print('Unsupported dataset: {}'.format(opt.dataset))

    if opt.dataset == 'RAVDESS':
        training_data = RAVDESS(
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            data_type='audiovisual',
            audio_transform=audio_transform
        )
    
    elif opt.dataset == 'CREMAD':
        training_data = CREMAD(
            annotation_path=opt.annotation_path,
            subset='training',
            spatial_transform=spatial_transform,
            audio_transform=audio_transform,
            data_type='audiovisual',
        )
    
    return training_data


def get_validation_set(opt, spatial_transform=None, audio_transform=None):
    """
    Get validation dataset based on specified dataset type
    Args:
        opt: options/configuration object with dataset parameters
        spatial_transform: video frame transformations
        audio_transform: audio transformations
    Returns:
        validation dataset
    """
    assert opt.dataset in ['RAVDESS', 'CREMAD'], \
        print('Unsupported dataset: {}'.format(opt.dataset))

    if opt.dataset == 'RAVDESS':
        validation_data = RAVDESS(
            opt.annotation_path,
            'validation',
            spatial_transform=spatial_transform,
            data_type='audiovisual',
            audio_transform=audio_transform
        )
    
    elif opt.dataset == 'CREMAD':
        validation_data = CREMAD(
            annotation_path=opt.annotation_path,
            subset='validation',
            spatial_transform=spatial_transform,
            audio_transform=audio_transform,
            data_type='audiovisual',
        )
    
    return validation_data


def get_test_set(opt, spatial_transform=None, audio_transform=None):
    """
    Get test dataset based on specified dataset type
    Args:
        opt: options/configuration object with dataset parameters
        spatial_transform: video frame transformations
        audio_transform: audio transformations
    Returns:
        test dataset
    """
    assert opt.dataset in ['RAVDESS', 'CREMAD'], \
        print('Unsupported dataset: {}'.format(opt.dataset))
    assert opt.test_subset in ['val', 'test'], \
        print('Unsupported test subset: {}'.format(opt.test_subset))

    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'
    
    if opt.dataset == 'RAVDESS':
        test_data = RAVDESS(
            opt.annotation_path,
            subset,
            spatial_transform=spatial_transform,
            data_type='audiovisual',
            audio_transform=audio_transform
        )
    
    elif opt.dataset == 'CREMAD':
        test_data = CREMAD(
            annotation_path=opt.annotation_path,
            subset=subset,
            spatial_transform=spatial_transform,
            audio_transform=audio_transform,
            data_type='audiovisual',
        )
    
    return test_data
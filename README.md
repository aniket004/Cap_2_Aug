# Cap2Aug: Caption guided Image data Augmentation
Aniket Roy, Anshul Shah∗, Ketul Shah∗, Anirban Roy, Rama Chellappa

Visual recognition in a low-data regime is challenging
and often prone to overfitting. To mitigate this issue, several
data augmentation strategies have been proposed. However,
standard transformations, e.g., rotation, cropping, and flip-
ping provide limited semantic variations. To this end, we
propose Cap2Aug, an image-to-image diffusion model-based
data augmentation strategy using image captions to condi-
tion the image synthesis step. We generate a caption for an
image and use this caption as an additional input for an
image-to-image diffusion model. This increases the semantic
diversity of the augmented images due to caption condition-
ing compared to the usual data augmentation techniques.
We show that Cap2Aug is particularly effective where only
a few samples are available for an object class. However,
naively generating the synthetic images is not adequate due
to the domain gap between real and synthetic images. Thus,
we employ a maximum mean discrepancy loss to align the
synthetic images to the real images to minimize the domain
gap. We evaluate our method on few-shot classification
and image classification with long-tail class distribution
tasks. Cap2Aug achieves state-of-the-art performance on
both tasks while evaluated on eleven benchmarks.

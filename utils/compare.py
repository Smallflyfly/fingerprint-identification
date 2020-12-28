from PIL import Image
from numpy import average, linalg, dot

def get_thumbnail(image, size=(1200, 750), greyscale=False):
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        image = image.convert('L')
    return image
def image_similarity_vectors_via_numpy(image1, image2):
    image1 = get_thumbnail(image1)
    image2 = get_thumbnail(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    res = dot(a / a_norm, b / b_norm)
    return res


def feature_compare(feature1, feature2):
    a_norm = linalg.norm(feature1, 2)
    b_norm = linalg.norm(feature2, 2)
    res = dot(feature1/a_norm, feature2/b_norm)
    return res
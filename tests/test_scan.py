import cv2
import numpy as np
from skimage.filters import threshold_otsu, threshold_sauvola

from docr import geometry, scan

rng = np.random.default_rng(1)


def text_page(angle=0.0):
    page = np.full((400, 300), 235, np.uint8)
    for row in range(40, 380, 28):
        cv2.putText(page, "docr scan test line", (20, row), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 30, 1)
    if angle:
        matrix = cv2.getRotationMatrix2D((150, 200), angle, 1.0)
        page = cv2.warpAffine(page, matrix, (300, 400), borderValue=235)
    return page


def test_otsu_matches_skimage():
    image = np.concatenate([rng.normal(60, 15, 3000), rng.normal(180, 20, 5000)]).clip(0, 255).astype(np.uint8)
    ours = image > scan.otsu(image)
    theirs = image > threshold_otsu(image)
    assert (ours == theirs).mean() > 0.999


def test_sauvola_matches_skimage():
    image = text_page().astype(np.float64)
    ours = scan.sauvola(image, window=25, k=0.2, r=128)
    theirs = threshold_sauvola(image, window_size=25, k=0.2, r=128)
    assert np.abs(ours - theirs).max() < 1e-6


def test_shading_flattening_removes_gradient():
    page = text_page().astype(np.float32)
    light = np.linspace(0.5, 1.0, page.shape[1])[None, :]
    flat = scan.flatten_shading(page * light)
    background = flat[page > 200]
    assert background.std() < 8


def test_skew_is_recovered():
    page = text_page(angle=2.5)
    assert abs(scan.estimate_skew(page) - (-2.5)) < 0.3


def test_page_is_found_on_table():
    table = np.zeros((480, 640, 3), np.uint8)
    table[:] = (70, 50, 40)
    paper = np.full((400, 300, 3), 240, np.uint8)
    target = np.array([[180, 60], [470, 90], [450, 440], [160, 410]], np.float64)
    matrix = geometry.homography(geometry.box_to_quad((0, 0, 299, 399)), target)
    warped = cv2.warpPerspective(paper, matrix, (640, 480))
    mask = cv2.warpPerspective(np.ones((400, 300), np.uint8), matrix, (640, 480)) > 0
    photo = np.where(mask[..., None], warped, table)
    quad, _ = scan.find_page(photo)
    assert scan.page_iou(quad, target, 640, 480) > 0.95


def test_douglas_peucker_keeps_corners():
    square = np.array([[0, 0], [5, 0.2], [10, 0], [10, 10], [5, 10.1], [0, 10]], np.float64)
    simple = scan.douglas_peucker(square, 0.5)
    assert len(simple) == 4

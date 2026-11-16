import cv2
import numpy as np
import pytest

from docr import edges, geometry, imaging

rng = np.random.default_rng(0)


def smooth_image(h=120, w=160):
    noise = rng.uniform(0, 255, (h // 8, w // 8)).astype(np.float32)
    return cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 255)


def page_image():
    image = np.full((200, 260), 230, np.uint8)
    cv2.rectangle(image, (30, 40), (220, 170), 40, 3)
    cv2.putText(image, "docr 42", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 20, 2)
    cv2.circle(image, (200, 60), 15, 90, -1)
    return image


def test_gray_matches_opencv():
    rgb = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
    ours = imaging.to_gray(rgb).astype(int)
    theirs = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(int)
    assert np.abs(ours - theirs).max() <= 1


@pytest.mark.parametrize("sigma,size", [(1.0, 7), (2.5, 15), (0.8, 5)])
def test_gaussian_matches_opencv(sigma, size):
    image = smooth_image()
    ours = imaging.gaussian_blur(image, sigma, size)
    theirs = cv2.GaussianBlur(image, (size, size), sigma, borderType=cv2.BORDER_REFLECT_101)
    assert np.abs(ours - theirs).max() < 1e-3


def test_sobel_matches_opencv():
    image = smooth_image()
    gx, gy = imaging.sobel(image)
    assert np.abs(gx - cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)).max() < 1e-3
    assert np.abs(gy - cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)).max() < 1e-3


@pytest.mark.parametrize("size", [(3, 3), (15, 7), (31, 31)])
def test_box_mean_matches_opencv(size):
    image = smooth_image()
    ours = imaging.box_mean(image, size)
    theirs = cv2.blur(image, (size[1], size[0]), borderType=cv2.BORDER_REFLECT_101)
    assert np.abs(ours - theirs).max() < 1e-3


@pytest.mark.parametrize("size", [(1, 9), (5, 5), (7, 13), (21, 3)])
def test_morphology_matches_opencv(size):
    image = rng.integers(0, 256, (60, 70), dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size[1], size[0]))
    assert np.array_equal(imaging.dilate(image, size), cv2.dilate(image, kernel))
    assert np.array_equal(imaging.erode(image, size), cv2.erode(image, kernel))


@pytest.mark.parametrize("connectivity", [4, 8])
def test_labels_match_opencv(connectivity):
    mask = rng.random((90, 110)) < 0.45
    ours, count = imaging.label(mask, connectivity)
    theirs_count, theirs = cv2.connectedComponents(mask.astype(np.uint8), connectivity=connectivity)
    assert count == theirs_count - 1
    pairs = np.unique(np.stack([ours[mask], theirs[mask]], axis=1), axis=0)
    assert len(pairs) == count


def test_component_stats():
    mask = np.zeros((20, 30), bool)
    mask[2:5, 3:9] = True
    mask[10:18, 20:22] = True
    labels, count = imaging.label(mask)
    areas, boxes = imaging.component_stats(labels, count)
    assert sorted(areas.tolist()) == [16, 18]
    assert [3, 2, 9, 5] in boxes.tolist()


def test_resize_matches_opencv_when_upscaling():
    image = smooth_image(30, 40)
    ours = imaging.resize(image, (75, 110))
    theirs = cv2.resize(image, (110, 75), interpolation=cv2.INTER_LINEAR)
    assert np.abs(ours - theirs).max() < 1e-2


def test_remap_matches_opencv():
    image = smooth_image()
    ys, xs = np.mgrid[0:100, 0:140].astype(np.float32)
    map_x = (xs * 1.1 + 3.3 + 4 * np.sin(ys / 9)).astype(np.float32)
    map_y = (ys * 0.9 + 7.7 + 3 * np.cos(xs / 11)).astype(np.float32)
    ours = imaging.remap(image, map_x, map_y, fill=0.0)
    theirs = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    inner = (map_x > 1) & (map_x < 158) & (map_y > 1) & (map_y < 118)
    assert np.abs(ours - theirs)[inner].max() < 1.5


def test_homography_matches_opencv():
    src = np.array([[10, 12], [200, 30], [190, 240], [15, 220]], np.float64)
    dst = np.array([[0, 0], [300, 0], [300, 400], [0, 400]], np.float64)
    ours = geometry.homography(src, dst)
    theirs = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    assert np.allclose(ours, theirs / theirs[2, 2], atol=1e-6)
    many = rng.uniform(0, 300, (30, 2))
    mapped = geometry.apply_homography(theirs, many) + rng.normal(0, 0.3, (30, 2))
    fitted = geometry.homography(many, mapped)
    reference, _ = cv2.findHomography(many, mapped, 0)
    probe = rng.uniform(0, 300, (10, 2))
    assert np.abs(geometry.apply_homography(fitted, probe) - geometry.apply_homography(reference, probe)).max() < 0.05


def test_warp_matches_opencv():
    image = smooth_image()
    matrix = np.array([[0.9, 0.1, 5.0], [-0.05, 1.05, 3.0], [1e-4, 2e-4, 1.0]])
    ours = geometry.warp_perspective(image, matrix, (150, 110))
    theirs = cv2.warpPerspective(image, matrix, (150, 110), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    assert np.mean(np.abs(ours - theirs)) < 0.5


def test_canny_matches_opencv():
    image = cv2.GaussianBlur(page_image(), (5, 5), 1.0)
    ours = edges.canny(image, 50, 150)
    theirs = cv2.Canny(image, 50, 150, L2gradient=True) > 0
    jaccard = (ours & theirs).sum() / (ours | theirs).sum()
    assert jaccard > 0.98


def test_hough_matches_opencv():
    image = np.zeros((160, 200), np.uint8)
    cv2.line(image, (10, 20), (190, 40), 255, 1)
    cv2.line(image, (50, 5), (60, 150), 255, 1)
    cv2.line(image, (0, 120), (199, 120), 255, 1)
    ours = edges.hough_lines(image > 0, threshold=60, max_lines=3)
    theirs = cv2.HoughLines(image, 1, np.pi / 180, 60)[:3, 0]
    assert np.allclose(ours[:, :2], theirs, atol=1e-4)


def test_polygon_iou_matches_opencv():
    for _ in range(20):
        a = geometry.convex_hull(rng.uniform(0, 100, (8, 2)))
        b = geometry.convex_hull(rng.uniform(20, 120, (8, 2)))
        theirs, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
        assert abs(geometry.intersection_area(a, b) - theirs) < 1e-2


def test_min_area_rect_matches_opencv():
    points = rng.normal(0, 1, (200, 2)) * [40, 8]
    t = np.deg2rad(25)
    points = points @ np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]]) + 100
    ours = geometry.area(geometry.min_area_rect(points))
    (_, (w, h), _) = cv2.minAreaRect(points.astype(np.float32))
    assert abs(ours - w * h) / (w * h) < 1e-3


def test_offset_convex_expands_square():
    square = geometry.box_to_quad((0, 0, 10, 10))
    grown = geometry.offset_convex(square, 2)
    assert np.allclose(geometry.area(grown), 14 * 14)
    shrunk = geometry.offset_convex(square, -2)
    assert np.allclose(geometry.area(shrunk), 6 * 6)


def test_crop_quad_reads_text_row():
    image = page_image().astype(np.float32)
    crop = geometry.crop_quad(image, geometry.box_to_quad((40, 80, 200, 120)), 32)
    assert crop.shape[0] == 32 and crop.shape[1] == 128

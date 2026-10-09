"""Tests de la fuente de datos de imágenes (`diffusion.data_generation.images`).

La suite es **autocontenida**: sintetiza imágenes con PIL en `tmp_path` (no depende
de `data/cats-prueba/`, que está gitignored y es local). Se omite entera (skip) si
faltan `torch`/`torchvision`, siguiendo la convención del repo. Todos los tests corren
con `num_workers=0` para estabilidad en CPU/Windows.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

# Si faltan las deps de imágenes, toda la suite se omite (7.3). El módulo bajo
# prueba se importa DESPUÉS de los importorskip.
torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

from PIL import Image

from diffusion.data_generation.images import (
    CatImages,
    _build_transform,
    count_images,
    finite_batches,
    infinite_batches,
    report_small_images,
)


# --------------------------------------------------------------- helpers/fixtures


def _save_rgb(path, width, height, rng):
    """Escribe un PNG RGB de ruido (uint8 en todo [0,255]) de `width`×`height`.

    El ruido garantiza píxeles oscuros y claros, así tras `Normalize` el batch
    cubre buena parte de [-1, 1] (permite chequear el rango de forma significativa).
    """
    arr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


@pytest.fixture
def rgb_dir(tmp_path):
    """Carpeta con 4 imágenes RGB sintéticas de tamaños variados (aspect ratios distintos)."""
    rng = np.random.default_rng(0)
    sizes = [(80, 100), (128, 64), (90, 90), (200, 150)]
    for i, (w, h) in enumerate(sizes):
        _save_rgb(tmp_path / f"img_{i}.png", w, h, rng)
    return tmp_path


@pytest.fixture
def rgb_dir_10(tmp_path):
    """Carpeta con 10 imágenes RGB sintéticas (nombres `img_00`…`img_09`, orden estable).

    Los nombres van con dos dígitos a propósito: así el orden lexicográfico del
    descubrimiento coincide con el numérico y los asserts sobre "las primeras N"
    se leen sin ambigüedad.
    """
    rng = np.random.default_rng(1)
    for i in range(10):
        _save_rgb(tmp_path / f"img_{i:02d}.png", 70 + i, 90 - i, rng)
    return tmp_path


# ------------------------------------------------------ contrato de salida (Req 1, 3, 5)


def test_contract_shape_dtype_range(rgb_dir):
    # (1.1, 1.2, 1.3, 3.1, 3.3, 5.1, 5.2) tensor crudo (B,3,64,64) float32 en [-1,1].
    it = infinite_batches(rgb_dir, 2, image_size=64, augment=False, num_workers=0, seed=0)
    batch = next(it)
    assert isinstance(batch, torch.Tensor)
    assert not isinstance(batch, tuple)  # tensor pelado, no una tupla (1.2)
    assert tuple(batch.shape) == (2, 3, 64, 64)
    assert batch.dtype == torch.float32
    assert torch.isfinite(batch).all()
    # Rango [-1,1] (3.3): la salida cae dentro del rango normalizado...
    assert batch.min().item() >= -1.0 - 1e-4
    assert batch.max().item() <= 1.0 + 1e-4
    # ...y hay valores < 0: sin `Normalize` la salida quedaría en [0,1] (todo >= 0),
    # así que este assert falla si se quitara la normalización a [-1,1].
    assert batch.min().item() < 0.0


def test_output_resized_to_image_size(rgb_dir):
    # (3.1) redimensiona a image_size×image_size sea cual sea el tamaño de entrada.
    it = infinite_batches(rgb_dir, 2, image_size=32, augment=False, num_workers=0, seed=0)
    assert tuple(next(it).shape) == (2, 3, 32, 32)


def test_infinite_does_not_exhaust(rgb_dir):
    # (1.4) 4 imgs / batch 2 = 2 batches por época; consumir 5 no levanta StopIteration.
    it = infinite_batches(rgb_dir, 2, image_size=64, augment=False, num_workers=0, seed=0)
    batches = list(itertools.islice(it, 5))
    assert len(batches) == 5
    for b in batches:
        assert tuple(b.shape) == (2, 3, 64, 64)


# ---------------------------------------------- descubrimiento y carga (Req 2)


def test_discovery_is_sorted(tmp_path):
    # (2.1) orden determinístico: las rutas descubiertas quedan ordenadas.
    rng = np.random.default_rng(0)
    for name in ("c.png", "a.png", "b.png"):
        _save_rgb(tmp_path / name, 80, 80, rng)
    ds = CatImages(tmp_path, _build_transform(64, augment=False))
    names = [p.name for p in ds.paths]
    assert names == sorted(names)
    assert len(ds) == 3


def test_non_rgb_yields_three_channels(tmp_path):
    # (2.4) grayscale + RGBA se convierten a 3 canales al cargarse. Con batch_size=2
    # y exactamente 2 imágenes no-RGB, el único batch DEBE contener ambas: si se
    # quitara `.convert("RGB")`, el collate fallaría (1 y 4 canales no apilan).
    Image.new("L", (80, 80), color=100).save(tmp_path / "gray.png")
    Image.new("RGBA", (80, 80), (10, 20, 30, 40)).save(tmp_path / "rgba.png")
    it = infinite_batches(tmp_path, 2, image_size=64, augment=False, num_workers=0, seed=0)
    batch = next(it)
    assert tuple(batch.shape) == (2, 3, 64, 64)
    assert batch.dtype == torch.float32


def test_cat_images_dataset_item(rgb_dir):
    # __getitem__ devuelve un tensor pelado (3,H,W) float32, sin label.
    ds = CatImages(rgb_dir, _build_transform(64, augment=False, crop=True))
    assert len(ds) == 4
    item = ds[0]
    assert isinstance(item, torch.Tensor)
    assert not isinstance(item, tuple)
    assert tuple(item.shape) == (3, 64, 64)
    assert item.dtype == torch.float32


# ---------------------------------------------------------- framing (Req 3.2)


@pytest.mark.parametrize("crop", [True, False])
def test_framing_crop_and_resize(rgb_dir, crop):
    # (3.2) center-crop (preserva aspect) y resize (deforma) ambos dan (B,3,64,64).
    it = infinite_batches(rgb_dir, 2, image_size=64, crop=crop, augment=False, num_workers=0, seed=0)
    assert tuple(next(it).shape) == (2, 3, 64, 64)


# ------------------------------------------------------- augmentation (Req 4)


def test_build_transform_flip_horizontal_only():
    # (4.2, 4.3, 4.4) con augment hay flip horizontal; sin augment no hay flip;
    # nunca flip vertical ni rotación (un gato al revés no es muestra válida).
    names_aug = [type(s).__name__ for s in _build_transform(64, augment=True, crop=True).transforms]
    names_no = [type(s).__name__ for s in _build_transform(64, augment=False, crop=True).transforms]
    assert "RandomHorizontalFlip" in names_aug
    assert "RandomHorizontalFlip" not in names_no
    for names in (names_aug, names_no):
        assert "RandomVerticalFlip" not in names
        assert "RandomRotation" not in names


def test_augment_off_is_deterministic(rgb_dir):
    # (4.1, 4.3) con augment=False + seed fijo, dos iteradores frescos dan el
    # mismo primer batch (sin flip aleatorio que rompa la reproducibilidad).
    it1 = infinite_batches(rgb_dir, 2, image_size=64, augment=False, num_workers=0, seed=0)
    it2 = infinite_batches(rgb_dir, 2, image_size=64, augment=False, num_workers=0, seed=0)
    assert torch.equal(next(it1), next(it2))


def test_augment_on_runs(rgb_dir):
    # (4.1) con augment=True el pipeline corre y respeta el contrato de salida.
    it = infinite_batches(rgb_dir, 2, image_size=64, augment=True, num_workers=0, seed=0)
    batch = next(it)
    assert tuple(batch.shape) == (2, 3, 64, 64)
    assert batch.dtype == torch.float32


# ----------------------------------------------- fail-fast (Req 2.2, 2.3, 5.2)


def test_root_missing_raises(tmp_path):
    # (2.2) carpeta raíz inexistente → ValueError, en la llamada (no en next()).
    with pytest.raises(ValueError):
        infinite_batches(tmp_path / "no_existe", 2, num_workers=0)


def test_empty_dir_raises(tmp_path):
    # (2.3) carpeta sin imágenes → ValueError (no dataset vacío silencioso).
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        infinite_batches(empty, 2, num_workers=0)


def test_batch_bigger_than_dataset_raises(tmp_path):
    # (5.2-guard) menos imágenes que batch_size → ValueError eager (no cuelga con
    # drop_last=True). Se levanta al construir el iterador, sin llamar a next().
    rng = np.random.default_rng(0)
    for i in range(3):
        _save_rgb(tmp_path / f"img_{i}.png", 80, 80, rng)
    with pytest.raises(ValueError):
        infinite_batches(tmp_path, 8, num_workers=0)


# --------------------------------------------------------- determinismo (Req 5.3)


def test_deterministic_sequence(rgb_dir):
    # (5.3) mismo seed (augment=False, num_workers=0) → misma secuencia de batches.
    it1 = infinite_batches(rgb_dir, 2, image_size=64, augment=False, num_workers=0, seed=7)
    it2 = infinite_batches(rgb_dir, 2, image_size=64, augment=False, num_workers=0, seed=7)
    for a, b in zip(itertools.islice(it1, 4), itertools.islice(it2, 4)):
        assert torch.equal(a, b)


# ------------------------------------------------- higiene report-only (Req 6)


def test_report_small_images_reports_short_side_and_keeps_files(tmp_path):
    # (6.1, 6.2) reporta por lado corto < min_size; report-only (no borra). El
    # criterio es min(width, height): 'wide' tiene lado corto 30 → reportada;
    # 'exact' tiene lado corto exactamente 64 → NO (no es < 64).
    rng = np.random.default_rng(0)
    _save_rgb(tmp_path / "tiny.png", 10, 10, rng)     # 10×10 → reportada
    _save_rgb(tmp_path / "wide.png", 100, 30, rng)    # lado corto 30 → reportada
    _save_rgb(tmp_path / "exact.png", 128, 64, rng)   # lado corto 64 → NO
    _save_rgb(tmp_path / "big.png", 128, 128, rng)    # ambos ≥ 64 → NO

    reported = {p.name for p in report_small_images(tmp_path, min_size=64)}
    assert "tiny.png" in reported
    assert "wide.png" in reported
    assert "exact.png" not in reported
    assert "big.png" not in reported

    # No borra ni modifica: todos los archivos siguen en disco.
    for name in ("tiny.png", "wide.png", "exact.png", "big.png"):
        assert (tmp_path / name).exists()


# ------------------------------- fuente finita (Req 2.2, 2.5, 2.7, 2.8, 3.9)


def test_finite_covers_whole_set_with_partial_tail(rgb_dir_10):
    # (2.7) 10 imágenes con batch 4 → 3 batches de 4/4/2 que suman 10: `drop_last=False`,
    # así que la cola del set NO se descarta (a diferencia de infinite_batches).
    batches = list(finite_batches(rgb_dir_10, 4, image_size=32))
    assert [b.shape[0] for b in batches] == [4, 4, 2]
    assert sum(b.shape[0] for b in batches) == 10
    for b in batches:
        assert isinstance(b, torch.Tensor)
        assert not isinstance(b, tuple)  # tensor pelado, no una tupla
        assert tuple(b.shape[1:]) == (3, 32, 32)
        assert b.dtype == torch.float32
        assert torch.isfinite(b).all()
        # Misma normalización a [-1, 1] que la fuente infinita.
        assert b.min().item() >= -1.0 - 1e-4
        assert b.max().item() <= 1.0 + 1e-4
    assert torch.cat(batches).min().item() < 0.0


def test_finite_is_reiterable(rgb_dir_10):
    # (2.2 / 3.3) el resultado es RE-ITERABLE, no un iterador de un solo uso:
    # dos recorridos completos entregan la misma secuencia de batches tensor por
    # tensor. Es la condición que hace reproducible el examen fijo.
    source = finite_batches(rgb_dir_10, 4, image_size=32)
    first = list(source)
    second = list(source)
    assert len(first) == len(second) == 3
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)


def test_finite_smaller_than_batch_yields_one_partial_batch(tmp_path):
    # (2.7) 3 imágenes con batch 8 → un único batch parcial de 3, SIN error: acá no
    # hay mínimo de batch_size (el de infinite_batches existe solo por drop_last=True).
    rng = np.random.default_rng(0)
    for i in range(3):
        _save_rgb(tmp_path / f"img_{i}.png", 80, 80, rng)
    batches = list(finite_batches(tmp_path, 8, image_size=32))
    assert len(batches) == 1
    assert tuple(batches[0].shape) == (3, 3, 32, 32)


def test_finite_order_and_transform_are_canonical(rgb_dir_10):
    # (2.2) el recorrido sigue el orden ordenado del descubrimiento y aplica la MISMA
    # cadena de transforms que la infinita con augment=False: concatenar los batches
    # reproduce exactamente los items de CatImages en orden. Falla si se barajara
    # (shuffle) o si la cadena incluyera el volteo aleatorio (orientación canónica).
    ds = CatImages(rgb_dir_10, _build_transform(32, augment=False, crop=True))
    expected = torch.stack([ds[i] for i in range(len(ds))])
    got = torch.cat(list(finite_batches(rgb_dir_10, 4, image_size=32, crop=True)))
    assert torch.equal(got, expected)


def test_finite_crop_false_frames_without_cropping(rgb_dir_10):
    # (2.2) el modo de encuadre se reusa tal cual: crop=False deforma al cuadrado y
    # coincide con la cadena equivalente aplicada a mano.
    ds = CatImages(rgb_dir_10, _build_transform(32, augment=False, crop=False))
    expected = torch.stack([ds[i] for i in range(len(ds))])
    got = torch.cat(list(finite_batches(rgb_dir_10, 5, image_size=32, crop=False)))
    assert torch.equal(got, expected)


def test_finite_max_images_takes_the_first_n_always(rgb_dir_10):
    # (2.8, 3.9) el tope entrega SIEMPRE las mismas N imágenes: las primeras N del
    # orden ordenado (subconjunto fijo, sin sorteo). Se compara contra el prefijo del
    # recorrido completo y contra un segundo recorrido de la fuente topeada.
    full = torch.cat(list(finite_batches(rgb_dir_10, 4, image_size=32)))
    capped = finite_batches(rgb_dir_10, 4, image_size=32, max_images=6)
    batches = list(capped)
    assert [b.shape[0] for b in batches] == [4, 2]  # 6 imágenes, cola parcial
    taken = torch.cat(batches)
    assert taken.shape[0] == 6
    assert torch.equal(taken, full[:6])
    # Re-iterar y reconstruir la fuente dan el mismo subconjunto.
    assert torch.equal(torch.cat(list(capped)), full[:6])
    otra = finite_batches(rgb_dir_10, 4, image_size=32, max_images=6)
    assert torch.equal(torch.cat(list(otra)), full[:6])


def test_finite_max_images_above_total_does_not_truncate(rgb_dir_10):
    # (2.8) un tope mayor que el set no recorta ni rompe: se entregan las 10 imágenes.
    # Es el caso de un set de entrenamiento más chico que el de validación.
    batches = list(finite_batches(rgb_dir_10, 4, image_size=32, max_images=99))
    assert sum(b.shape[0] for b in batches) == 10


def test_finite_does_not_consume_global_rng(rgb_dir_10):
    # (invariante) sin barajado ni augmentation la fuente no tiene aleatoriedad, así
    # que no puede consumir el RNG global: activar la validación no debe mover el
    # azar del entrenamiento.
    before = torch.random.get_rng_state()
    list(finite_batches(rgb_dir_10, 4, image_size=32))
    assert torch.equal(torch.random.get_rng_state(), before)


def test_finite_root_missing_raises(tmp_path):
    # (2.5) carpeta raíz inexistente → ValueError en la llamada (fail-fast heredado
    # de CatImages, antes de devolver la fuente).
    with pytest.raises(ValueError):
        finite_batches(tmp_path / "no_existe", 2)


def test_finite_empty_dir_raises(tmp_path):
    # (2.5) carpeta sin imágenes → ValueError (no una fuente vacía silenciosa).
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        finite_batches(empty, 2)


# ----------------------------------------------- conteo de imágenes (Req 2.8)


def test_count_images_matches_the_real_amount(rgb_dir_10):
    # (2.8) el conteo coincide con la cantidad real de imágenes de la carpeta, que es
    # lo que después dimensiona el examen de entrenamiento.
    assert count_images(rgb_dir_10) == 10
    assert count_images(rgb_dir_10) == sum(
        b.shape[0] for b in finite_batches(rgb_dir_10, 4, image_size=32)
    )


def test_count_images_is_recursive_and_ignores_non_images(tmp_path):
    # (2.8) misma definición de "imagen" que la fuente: recursivo por subcarpetas y
    # filtrado por extensión (un .txt no cuenta).
    rng = np.random.default_rng(0)
    _save_rgb(tmp_path / "a.png", 80, 80, rng)
    sub = tmp_path / "sub"
    sub.mkdir()
    _save_rgb(sub / "b.png", 80, 80, rng)
    (tmp_path / "notas.txt").write_text("no soy una imagen", encoding="utf-8")
    assert count_images(tmp_path) == 2


def test_count_images_missing_root_raises(tmp_path):
    # (2.5) mismo fail-fast que la fuente: raíz inexistente → ValueError.
    with pytest.raises(ValueError):
        count_images(tmp_path / "no_existe")


def test_count_images_empty_dir_raises(tmp_path):
    # (2.5) mismo fail-fast que la fuente: carpeta sin imágenes → ValueError.
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        count_images(empty)


def test_finite_and_count_are_exported_by_the_package(rgb_dir_10):
    # Los dos símbolos se importan del paquete igual que infinite_batches: es la
    # superficie que consume el config layer.
    from diffusion.data_generation import count_images as pkg_count
    from diffusion.data_generation import finite_batches as pkg_finite

    assert pkg_count(rgb_dir_10) == 10
    assert sum(b.shape[0] for b in pkg_finite(rgb_dir_10, 4, image_size=32)) == 10

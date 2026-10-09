"""Tests de la herramienta de split determinístico del dataset (``scripts/split_dataset.py``).

El script no es un módulo del paquete ``diffusion``: se carga **por ruta** con ``importlib``, la
convención ya usada para ``scripts/train.py`` y ``scripts/sample.py`` en ``test_resume.py`` y
``test_samplers.py``.

Esta suite corre sin torch: el reparto (``split_paths``) es una función pura de la stdlib y la
E/S usa solo ``pathlib``/``shutil``. Las imágenes de los tests de E/S se sintetizan con PIL en
``tmp_path`` (el patrón de ``tests/test_image_data.py``), así la suite no depende de
``data/cats-prueba/``.
"""

from __future__ import annotations

import importlib.util
import pathlib
import random

import pytest

from PIL import Image

from diffusion.data_generation.images import IMAGE_EXTENSIONS

_SPLIT_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "split_dataset.py"


def _load_split_script():
    """Carga ``scripts/split_dataset.py`` por ruta y devuelve el módulo."""
    spec = importlib.util.spec_from_file_location("_split_dataset_under_test", _SPLIT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _paths(n: int, root: str = "data/cats") -> list[pathlib.Path]:
    """``n`` rutas sintéticas, **ya ordenadas** (precondición de ``split_paths``)."""
    return sorted(pathlib.Path(root) / f"{i:05d}.jpg" for i in range(n))


def _crear_imagenes(root: pathlib.Path, nombres: list[str]) -> list[str]:
    """Escribe una imagen RGB mínima por cada nombre relativo (creando subcarpetas).

    Args:
        root: Carpeta base bajo la que se escriben los archivos.
        nombres: Rutas relativas (con ``/`` como separador) de las imágenes a crear.

    Returns:
        La lista de nombres recibida, para poder compararla con lo que quedó en el destino.
    """
    for i, nombre in enumerate(nombres):
        destino = root / nombre
        destino.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), color=(i % 256, 7, 11)).save(destino)
    return nombres


def _relativos(root: pathlib.Path) -> list[str]:
    """Rutas relativas (con ``/``) de todos los archivos bajo ``root``, ordenadas."""
    if not root.exists():
        return []
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


# =============================================================================
# task 1.1 — Reparto determinístico como función pura
# =============================================================================
#
# Contrato (design, "Components and Interfaces → Herramienta → DatasetSplitter"):
#     split_paths(paths, val_frac, seed) -> (train, val)
# Precondiciones: ``paths`` no vacío y ya ordenado; 0 < val_frac < 1.
# Postcondiciones: len(val) == round(len(paths) * val_frac) (1.2); (train, val) es una partición
# de ``paths`` con ambas partes no vacías; el resultado depende solo de (paths, val_frac, seed).


# ---------------------------------------------------------------- determinismo (1.3, 1.4)


def test_misma_seed_da_particion_identica():
    # 1.3: dos llamadas con la misma seed, el mismo frac y el mismo conjunto ⇒ mismo reparto.
    mod = _load_split_script()
    paths = _paths(50)

    train_a, val_a = mod.split_paths(paths, 0.2, 7)
    train_b, val_b = mod.split_paths(paths, 0.2, 7)

    assert train_a == train_b
    assert val_a == val_b


def test_seeds_distintas_dan_particiones_distintas():
    # 1.4: con al menos 20 archivos, dos seeds distintas ⇒ particiones distintas.
    mod = _load_split_script()
    paths = _paths(20)

    _, val_0 = mod.split_paths(paths, 0.3, 0)
    _, val_1 = mod.split_paths(paths, 0.3, 1)

    assert set(val_0) != set(val_1)


def test_no_usa_el_rng_global():
    # 1.3 (mecanismo): el azar sale de un RNG sembrado propio, así que el estado del RNG global
    # no se mueve y el reparto no depende de quién sorteó antes.
    mod = _load_split_script()
    paths = _paths(30)

    random.seed(1234)
    estado_previo = random.getstate()
    train, val = mod.split_paths(paths, 0.25, 3)
    assert random.getstate() == estado_previo

    # Y el reparto es el mismo aunque el RNG global esté en otro estado.
    random.seed(9999)
    assert (train, val) == mod.split_paths(paths, 0.25, 3)


def test_no_muta_la_lista_de_entrada():
    # El reparto es una función pura: la lista recibida queda intacta.
    mod = _load_split_script()
    paths = _paths(12)
    copia = list(paths)

    mod.split_paths(paths, 0.25, 0)

    assert paths == copia


# ---------------------------------------------------------------- partición y conteo (1.2)


@pytest.mark.parametrize(
    ("n", "val_frac"),
    [(10, 0.1), (10, 0.2), (100, 0.1), (37, 0.15), (20, 0.5), (7, 0.3), (999, 0.05)],
)
def test_conteo_de_val_por_redondeo(n, val_frac):
    # 1.2: a val le toca la cantidad que resulta de redondear el porcentaje sobre el total,
    # y el resto a train.
    mod = _load_split_script()
    paths = _paths(n)

    train, val = mod.split_paths(paths, val_frac, 0)

    assert len(val) == round(n * val_frac)
    assert len(train) == n - len(val)


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_union_reconstruye_la_entrada_sin_interseccion(seed):
    # 1.1 (parte pura): (train, val) es una partición del conjunto de entrada — ningún archivo
    # se duplica y ninguno se pierde.
    mod = _load_split_script()
    paths = _paths(40)

    train, val = mod.split_paths(paths, 0.25, seed)

    assert set(train) | set(val) == set(paths)
    assert not set(train) & set(val)
    assert len(train) + len(val) == len(paths)
    assert train and val


def test_reparto_de_rutas_con_subcarpetas():
    # Las rutas son opacas para el reparto: un dataset con subcarpetas se reparte igual.
    mod = _load_split_script()
    paths = sorted(
        [pathlib.Path("src") / "a" / f"{i}.jpg" for i in range(10)]
        + [pathlib.Path("src") / "b" / f"{i}.png" for i in range(10)]
    )

    train, val = mod.split_paths(paths, 0.2, 5)

    assert len(val) == 4
    assert set(train) | set(val) == set(paths)
    assert all(isinstance(p, pathlib.Path) for p in train + val)


# ---------------------------------------------------------------- rechazos (1.9, 1.10)


@pytest.mark.parametrize("val_frac", [0.0, 1.0, -0.1, 1.5, 2.0])
def test_val_frac_fuera_del_intervalo_abierto_falla(val_frac):
    # 1.9: fuera del intervalo abierto (0, 1) ⇒ ValueError informando el valor recibido.
    mod = _load_split_script()
    paths = _paths(20)

    with pytest.raises(ValueError) as exc:
        mod.split_paths(paths, val_frac, 0)

    assert repr(val_frac) in str(exc.value) or str(val_frac) in str(exc.value)


def test_val_vacia_falla_informando_total_y_porcentaje():
    # 1.10: 3 imágenes con 10% ⇒ round(0.3) == 0 ⇒ val quedaría vacía.
    mod = _load_split_script()

    with pytest.raises(ValueError) as exc:
        mod.split_paths(_paths(3), 0.1, 0)

    mensaje = str(exc.value)
    assert "3" in mensaje
    assert "0.1" in mensaje


def test_train_vacia_falla_informando_total_y_porcentaje():
    # 1.10: 2 imágenes con 90% ⇒ round(1.8) == 2 ⇒ train quedaría vacía.
    mod = _load_split_script()

    with pytest.raises(ValueError) as exc:
        mod.split_paths(_paths(2), 0.9, 0)

    mensaje = str(exc.value)
    assert "2" in mensaje
    assert "0.9" in mensaje


def test_lista_vacia_falla():
    # Precondición "paths no vacío": sin archivos no hay reparto posible (1.10/1.11).
    mod = _load_split_script()

    with pytest.raises(ValueError):
        mod.split_paths([], 0.2, 0)


# =============================================================================
# task 1.2 — Script ejecutable con la E/S y las validaciones
# =============================================================================
#
# Contrato (design, "Components and Interfaces → Herramienta → DatasetSplitter → Batch / Job
# Contract"):
#     python scripts/split_dataset.py --src S --out O --val-frac F --seed N [--move] [--overwrite]
# Salida: ``<out>/train/<relpath>`` y ``<out>/val/<relpath>`` + conteos por consola (1.1, 1.7).
# Errores: ValueError mapeado a código de salida 2 con el mensaje en stderr (1.8, 1.9, 1.10, 1.11).
# Todas las validaciones corren **antes** de tocar el primer archivo: un error no deja un destino
# a medias.


def _argv(src, out, val_frac=0.2, seed=0, extra=()):
    """Arma la lista de argumentos del CLI (rutas como str, como las pasa la consola)."""
    return [
        "--src", str(src),
        "--out", str(out),
        "--val-frac", str(val_frac),
        "--seed", str(seed),
        *extra,
    ]


# ---------------------------------------------------------------- reparto completo (1.1, 1.7)


def test_escribe_train_y_val_con_la_particion_completa(tmp_path):
    # 1.1: dos carpetas bajo el destino que reparten TODOS los archivos del origen, sin duplicar.
    mod = _load_split_script()
    src, out = tmp_path / "src", tmp_path / "out"
    nombres = _crear_imagenes(src, [f"img_{i:02d}.png" for i in range(10)])

    assert mod.main(_argv(src, out)) == 0

    en_train, en_val = _relativos(out / "train"), _relativos(out / "val")
    assert len(en_val) == 2  # round(10 * 0.2)
    assert len(en_train) == 8
    assert sorted(en_train + en_val) == sorted(nombres)
    assert not set(en_train) & set(en_val)


def test_informa_los_conteos_por_consola(tmp_path, capsys):
    # 1.7: al terminar informa cuántas imágenes quedaron en cada partición.
    mod = _load_split_script()
    src, out = tmp_path / "src", tmp_path / "out"
    _crear_imagenes(src, [f"img_{i:02d}.png" for i in range(10)])

    assert mod.main(_argv(src, out)) == 0

    salida = capsys.readouterr().out
    assert "train" in salida and "val" in salida
    # Los conteos se buscan en su forma renderizada ("(8 imágenes)"), no como dígitos sueltos: un
    # "2" pelado ya lo satisface el val_frac=0.2 que se imprime más arriba, así que no probaría nada.
    assert "(8 imágenes)" in salida
    assert "(2 imágenes)" in salida


def test_particion_identica_entre_dos_corridas_con_la_misma_seed(tmp_path):
    # 1.3 de punta a punta: el determinismo sobrevive al descubrimiento en disco.
    mod = _load_split_script()
    src = tmp_path / "src"
    _crear_imagenes(src, [f"img_{i:02d}.png" for i in range(20)])

    assert mod.main(_argv(src, tmp_path / "a", val_frac=0.25, seed=3)) == 0
    assert mod.main(_argv(src, tmp_path / "b", val_frac=0.25, seed=3)) == 0

    assert _relativos(tmp_path / "a" / "val") == _relativos(tmp_path / "b" / "val")
    assert _relativos(tmp_path / "a" / "train") == _relativos(tmp_path / "b" / "train")


# ---------------------------------------------------------------- copia vs movimiento (1.5, 1.6)


def test_por_defecto_copia_y_deja_el_origen_intacto(tmp_path):
    # 1.5: sin --move el origen queda igual: los 10 archivos siguen ahí.
    mod = _load_split_script()
    src, out = tmp_path / "src", tmp_path / "out"
    nombres = _crear_imagenes(src, [f"img_{i:02d}.png" for i in range(10)])

    assert mod.main(_argv(src, out)) == 0

    assert _relativos(src) == sorted(nombres)


def test_move_mueve_los_archivos_y_vacia_el_origen(tmp_path):
    # 1.6: con --move los archivos se mueven en vez de copiarse.
    mod = _load_split_script()
    src, out = tmp_path / "src", tmp_path / "out"
    nombres = _crear_imagenes(src, [f"img_{i:02d}.png" for i in range(10)])

    assert mod.main(_argv(src, out, extra=["--move"])) == 0

    assert _relativos(src) == []
    assert sorted(_relativos(out / "train") + _relativos(out / "val")) == sorted(nombres)


# ---------------------------------------------------------------- subcarpetas (1.1)


def test_preserva_la_ruta_relativa_y_evita_colisiones_de_nombre(tmp_path):
    # 1.1: dos subcarpetas con los MISMOS nombres de archivo no pueden pisarse en el destino.
    mod = _load_split_script()
    src, out = tmp_path / "src", tmp_path / "out"
    nombres = _crear_imagenes(
        src,
        [f"a/img_{i:02d}.png" for i in range(5)] + [f"b/img_{i:02d}.png" for i in range(5)],
    )

    assert mod.main(_argv(src, out)) == 0

    replicados = _relativos(out / "train") + _relativos(out / "val")
    assert sorted(replicados) == sorted(nombres)  # 10 archivos, ninguno colisionó
    assert all("/" in rel for rel in replicados)  # cada uno bajo su subcarpeta


def test_usa_la_misma_definicion_de_imagen_que_el_loader(tmp_path):
    # Corrección (design): el split reparte exactamente el conjunto que el loader lee, así que
    # comparte la constante pública en vez de duplicar la lista de extensiones. Los archivos que
    # no son imagen quedan afuera del reparto.
    mod = _load_split_script()
    src, out = tmp_path / "src", tmp_path / "out"
    _crear_imagenes(src, [f"img_{i:02d}.png" for i in range(10)])
    (src / "notas.txt").write_text("no es una imagen", encoding="utf-8")
    (src / "labels.csv").write_text("tampoco", encoding="utf-8")

    assert mod.IMAGE_EXTENSIONS is IMAGE_EXTENSIONS
    assert mod.main(_argv(src, out)) == 0

    replicados = _relativos(out / "train") + _relativos(out / "val")
    assert len(replicados) == 10
    assert not any(rel.endswith((".txt", ".csv")) for rel in replicados)


def test_descubre_extensiones_en_mayusculas(tmp_path):
    # La comparación de extensiones es en minúsculas (igual que el loader): .JPG cuenta como imagen.
    mod = _load_split_script()
    src, out = tmp_path / "src", tmp_path / "out"
    nombres = _crear_imagenes(src, [f"img_{i:02d}.JPG" for i in range(10)])

    assert mod.main(_argv(src, out)) == 0

    assert sorted(_relativos(out / "train") + _relativos(out / "val")) == sorted(nombres)


# ---------------------------------------------------------------- destino ocupado (1.8)


def test_destino_ocupado_sin_overwrite_aborta_sin_escribir(tmp_path, capsys):
    # 1.8: si el destino ya tiene contenido y no se pidió sobreescribir, aborta informando la
    # ruta en conflicto y sin escribir nada.
    mod = _load_split_script()
    src, out = tmp_path / "src", tmp_path / "out"
    _crear_imagenes(src, [f"img_{i:02d}.png" for i in range(10)])
    _crear_imagenes(out / "train", ["viejo.png"])

    assert mod.main(_argv(src, out)) == 2

    err = capsys.readouterr().err
    assert "train" in err
    assert _relativos(out / "train") == ["viejo.png"]  # no se copió nada encima
    assert not (out / "val").exists()


def test_destino_val_ocupado_sin_overwrite_aborta(tmp_path, capsys):
    # 1.8: la validación mira las dos carpetas del destino, no solo train/.
    mod = _load_split_script()
    src, out = tmp_path / "src", tmp_path / "out"
    _crear_imagenes(src, [f"img_{i:02d}.png" for i in range(10)])
    _crear_imagenes(out / "val", ["sub/viejo.png"])

    assert mod.main(_argv(src, out)) == 2

    assert "val" in capsys.readouterr().err
    assert not (out / "train").exists()


def test_destino_vacio_no_es_conflicto(tmp_path):
    # 1.8: "contiene archivos" — una carpeta de destino existente pero vacía no bloquea el split.
    mod = _load_split_script()
    src, out = tmp_path / "src", tmp_path / "out"
    _crear_imagenes(src, [f"img_{i:02d}.png" for i in range(10)])
    (out / "train").mkdir(parents=True)
    (out / "val").mkdir(parents=True)

    assert mod.main(_argv(src, out)) == 0

    assert len(_relativos(out / "train") + _relativos(out / "val")) == 10


def test_overwrite_permite_el_destino_ocupado(tmp_path):
    # 1.8 (complemento): con --overwrite el split procede sobre el destino ocupado.
    mod = _load_split_script()
    src, out = tmp_path / "src", tmp_path / "out"
    nombres = _crear_imagenes(src, [f"img_{i:02d}.png" for i in range(10)])
    _crear_imagenes(out / "train", ["img_00.png"])  # mismo nombre que una del origen

    assert mod.main(_argv(src, out, extra=["--overwrite"])) == 0

    assert sorted(_relativos(out / "train") + _relativos(out / "val")) == sorted(nombres)


def test_overwrite_limpia_el_destino_y_no_deja_imagenes_en_las_dos_particiones(tmp_path):
    # 1.1 + 1.7 (regresión): re-partir con OTRA seed es el motivo habitual para re-correr la
    # herramienta. Si --overwrite escribiera encima sin limpiar, las imágenes que cambiaron de
    # lado quedarían en train/ **y** en val/ a la vez (una imagen entrenada y contada como
    # held-out), y los conteos informados dejarían de describir lo que hay en disco.
    mod = _load_split_script()
    src, out = tmp_path / "src", tmp_path / "out"
    nombres = _crear_imagenes(src, [f"img_{i:02d}.png" for i in range(20)])

    assert mod.main(_argv(src, out, val_frac=0.2, seed=0)) == 0
    assert mod.main(_argv(src, out, val_frac=0.2, seed=7, extra=["--overwrite"])) == 0

    en_train, en_val = _relativos(out / "train"), _relativos(out / "val")
    assert not set(en_train) & set(en_val)  # particiones disjuntas
    assert sorted(en_train + en_val) == sorted(nombres)  # el total es el del origen
    assert len(en_val) == 4 and len(en_train) == 16  # y coincide con los conteos informados


def test_overwrite_borra_los_sobrantes_del_split_anterior(tmp_path):
    # 1.1: la limpieza no distingue procedencias — cualquier archivo previo bajo train/ o val/
    # desaparece, así el destino describe solo el reparto de esta corrida.
    mod = _load_split_script()
    src, out = tmp_path / "src", tmp_path / "out"
    nombres = _crear_imagenes(src, [f"img_{i:02d}.png" for i in range(10)])
    _crear_imagenes(out / "train", ["ajeno.png", "sub/otro.png"])
    _crear_imagenes(out / "val", ["ajeno.png"])

    assert mod.main(_argv(src, out, extra=["--overwrite"])) == 0

    assert sorted(_relativos(out / "train") + _relativos(out / "val")) == sorted(nombres)


def test_overwrite_no_borra_nada_si_una_validacion_falla(tmp_path, capsys):
    # Invariante de orden (design): la limpieza ocurre DESPUÉS de todas las validaciones, así un
    # error nunca deja el destino peor de como estaba (acá: val_frac inválido con --overwrite).
    mod = _load_split_script()
    src, out = tmp_path / "src", tmp_path / "out"
    _crear_imagenes(src, [f"img_{i:02d}.png" for i in range(10)])
    _crear_imagenes(out / "train", ["viejo.png"])

    assert mod.main(_argv(src, out, val_frac=1.5, extra=["--overwrite"])) == 2

    assert "1.5" in capsys.readouterr().err
    assert _relativos(out / "train") == ["viejo.png"]  # intacto


# ------------------------------------------- origen y destino solapados (precondición del borrado)


def test_destino_igual_al_origen_aborta_sin_tocar_nada(tmp_path, capsys):
    # Precondición del borrado de --overwrite: limpiar '<out>/train' cuando el destino ES el origen
    # borraría imágenes originales. Se aborta antes de escribir y antes de borrar.
    mod = _load_split_script()
    src = tmp_path / "cats"
    nombres = _crear_imagenes(src, [f"train/img_{i:02d}.png" for i in range(10)])

    assert mod.main(_argv(src, src, extra=["--overwrite"])) == 2

    assert "error:" in capsys.readouterr().err
    assert _relativos(src) == sorted(nombres)  # las originales siguen ahí


def test_destino_anidado_en_el_origen_aborta(tmp_path, capsys):
    # El destino dentro del origen también rompe el descubrimiento: la búsqueda recursiva barrería
    # las copias de la corrida anterior y repartiría más imágenes de las que hay.
    mod = _load_split_script()
    src = tmp_path / "src"
    nombres = _crear_imagenes(src, [f"img_{i:02d}.png" for i in range(10)])

    assert mod.main(_argv(src, src / "split")) == 2

    assert "error:" in capsys.readouterr().err
    assert not (src / "split").exists()
    assert _relativos(src) == sorted(nombres)


def test_origen_anidado_en_el_destino_aborta(tmp_path, capsys):
    # El caso simétrico: limpiar el destino borraría el origen entero.
    mod = _load_split_script()
    out = tmp_path / "out"
    src = out / "src"
    nombres = _crear_imagenes(src, [f"img_{i:02d}.png" for i in range(10)])

    assert mod.main(_argv(src, out, extra=["--overwrite"])) == 2

    assert "error:" in capsys.readouterr().err
    assert _relativos(src) == sorted(nombres)


def test_rutas_solapadas_escritas_con_punto_y_dosdos_tambien_se_detectan(tmp_path, capsys):
    # Las rutas se resuelven antes de compararlas, así que '.' / '..' no burlan la precondición.
    mod = _load_split_script()
    src = tmp_path / "src"
    _crear_imagenes(src, [f"img_{i:02d}.png" for i in range(10)])
    disfrazado = src / "sub" / ".." / "."

    assert mod.main(_argv(src, disfrazado, extra=["--overwrite"])) == 2

    assert "error:" in capsys.readouterr().err
    assert not (src / "train").exists()


# ---------------------------------------------------------------- fallas de E/S


def test_error_de_escritura_se_informa_como_error_y_avisa_del_destino_incompleto(
    tmp_path, capsys, monkeypatch
):
    # Patrón de la casa: un fallo de E/S (permisos, disco lleno, cross-device) no puede salir como
    # traceback. Se informa 'error: ...' en stderr con código 2, avisando que el destino puede
    # haber quedado a medias (es el único camino que sí lo deja incompleto).
    mod = _load_split_script()
    src, out = tmp_path / "src", tmp_path / "out"
    _crear_imagenes(src, [f"img_{i:02d}.png" for i in range(10)])

    def _falla(*args, **kwargs):
        raise OSError("no queda espacio en el dispositivo")

    monkeypatch.setattr(mod.shutil, "copy2", _falla)

    assert mod.main(_argv(src, out)) == 2

    err = capsys.readouterr().err
    assert "error:" in err
    assert "no queda espacio en el dispositivo" in err
    assert "incompleto" in err


# ---------------------------------------------------------------- origen inválido (1.11)


def test_origen_inexistente_aborta_informando_la_ruta(tmp_path, capsys):
    # 1.11: carpeta de origen inexistente ⇒ código de salida != 0 y la ruta en el mensaje.
    mod = _load_split_script()
    src, out = tmp_path / "no_existe", tmp_path / "out"

    assert mod.main(_argv(src, out)) == 2

    assert "no_existe" in capsys.readouterr().err
    assert not out.exists()


def test_origen_sin_imagenes_aborta_informando_la_ruta(tmp_path, capsys):
    # 1.11: carpeta existente pero vacía ⇒ aborta informando la ruta.
    mod = _load_split_script()
    src, out = tmp_path / "vacia", tmp_path / "out"
    src.mkdir()

    assert mod.main(_argv(src, out)) == 2

    assert "vacia" in capsys.readouterr().err
    assert not out.exists()


def test_origen_solo_con_archivos_que_no_son_imagenes_aborta(tmp_path, capsys):
    # 1.11: "sin imágenes" se juzga con la definición del loader, no con "carpeta vacía".
    mod = _load_split_script()
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    (src / "notas.txt").write_text("x", encoding="utf-8")

    assert mod.main(_argv(src, out)) == 2

    assert "src" in capsys.readouterr().err
    assert not out.exists()


# ---------------------------------------------------------------- rechazos mapeados a exit 2


def test_val_frac_invalido_devuelve_codigo_2_sin_escribir(tmp_path, capsys):
    # 1.9 vía main: el ValueError del reparto se mapea a código 2 con el mensaje en stderr.
    mod = _load_split_script()
    src, out = tmp_path / "src", tmp_path / "out"
    _crear_imagenes(src, [f"img_{i:02d}.png" for i in range(10)])

    assert mod.main(_argv(src, out, val_frac=1.5)) == 2

    assert "1.5" in capsys.readouterr().err
    assert not out.exists()


def test_reparto_que_dejaria_una_parte_vacia_devuelve_codigo_2_sin_escribir(tmp_path, capsys):
    # 1.10 vía main: 3 imágenes con 10% ⇒ val vacía ⇒ aborta antes de tocar el primer archivo.
    mod = _load_split_script()
    src, out = tmp_path / "src", tmp_path / "out"
    _crear_imagenes(src, [f"img_{i:02d}.png" for i in range(3)])

    assert mod.main(_argv(src, out, val_frac=0.1)) == 2

    err = capsys.readouterr().err
    assert "3" in err and "0.1" in err
    assert not out.exists()
    assert len(_relativos(src)) == 3  # el origen sigue intacto


def test_help_imprime_uso_sin_error(capsys):
    # El CLI sigue el patrón de la casa: argparse + main(argv) -> int.
    mod = _load_split_script()

    with pytest.raises(SystemExit) as exc:
        mod.main(["--help"])

    assert exc.value.code == 0
    assert "--val-frac" in capsys.readouterr().out

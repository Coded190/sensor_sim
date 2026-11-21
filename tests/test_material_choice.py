"""BSDF choice follows the source behavior. pplastic is not the default for every surface."""

from sensor_sim.scene_gen.material_choice import choose_material


def test_matte_surface_is_depolarizing_diffuse():
    choice = choose_material({"name": "plaster", "roughness": 0.9, "metallic": 0.0, "transmission": 0.0})
    assert choice["family"] == "diffuse"
    assert choice["polarization"] == "depolarizing"
    assert choice["quantitative_polarization"] is False


def test_coated_surface_uses_pplastic_as_an_approximation():
    choice = choose_material({"name": "sofa_fabric", "roughness": 0.4, "metallic": 0.0, "transmission": 0.0})
    assert choice["family"] == "pplastic"
    assert "not energy conserving" in choice["note"]
    assert choice["quantitative_polarization"] is False


def test_thin_glass_is_not_relabeled_as_solid_dielectric():
    choice = choose_material({"name": "glass_pane", "roughness": 0.02, "transmission": 1.0, "thin": True})
    assert choice["family"] == "thindielectric"
    assert choice["polarization"] == "depolarizing"


def test_rough_glass_stays_rough():
    choice = choose_material({"name": "glass_frosted", "roughness": 0.4, "transmission": 1.0, "thin": False})
    assert choice["family"] == "roughdielectric"


def test_metalness_does_not_invent_conductor_constants():
    choice = choose_material({"name": "chrome_handle", "metallic": 0.9, "roughness": 0.2})
    assert choice["family"] == "diffuse"
    assert "metallic_optical_constants" in choice["lost"]


def test_measured_pbrdf_is_used_only_when_supplied():
    choice = choose_material({"name": "fabric", "measured_pbrdf": True, "roughness": 0.4})
    assert choice["family"] == "measured_polarized"

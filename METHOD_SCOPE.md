# Method scope

Primary method: one native spline optimization initialized by the frozen network. The code distinguishes geometry validation, execution scheduling and planning, and does not establish hard real-time or hardware performance.

Primary source entries:

- `scripts/audit_curobo_native.py`
- `scripts/audit_curobo_spheres.py`
- `scripts/audit_moving_spline_boundary.py`
- `scripts/audit_moving_spline_development.py`
- `scripts/audit_native_spline_adjoint.py`
- `scripts/audit_native_spline_sampling.py`
- `scripts/audit_normalized_spline_adapter.py`
- `scripts/audit_online_spline_handoff.py`
- `scripts/audit_portable_fcl.py`
- `scripts/audit_spline_repeat.py`
- `scripts/build_curobo_scene.py`
- `scripts/build_spline_seed_encoder.py`
- `scripts/compare_native_splines.py`
- `scripts/compare_paper_environment_splines.py`
- `scripts/evaluate_curobo_development.py`
- `scripts/evaluate_exact_spline.py`
- `scripts/export_portable_scene.py`
- `scripts/freeze_spline_confirmation.py`
- `scripts/generate_portable_holdout.py`
- `scripts/inspect_spline_delivery_geometry.py`
- `scripts/render_simple_spline_video.py`
- `scripts/report_paper_matched_planning.py`
- `scripts/report_spline_confirmation.py`
- `scripts/summarize_online_spline_audit.py`
- `scripts/summarize_spline_development.py`
- `scripts/test_curobo_mesh_regression.py`
- `scripts/verify_simple_spline_video.py`
- `src/planning/exact_spline.py`
- `src/planning/kan_seed.py`
- `src/planning/native_spline_sampling.py`
- `src/planning/normalized_spline_objective.py`
- `src/planning/normalized_spline_planner.py`
- `src/planning/spline_seed_encoding.py`
- `src/planning/tcp_spline_objective.py`

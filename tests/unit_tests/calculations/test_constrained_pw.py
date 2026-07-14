"""Tests for the ``ConstrainedPWCalculation`` class."""

import numpy as np
import pytest


class TestWriteOscdftData:
    """Test the ``write_oscdft_data`` method directly."""

    def test_formatting_basic(self):
        """Test that write_oscdft_data produces correctly formatted output."""
        from lordcapulet.calculations.constrained_pw import ConstrainedPWCalculation

        parameters = {
            'nconstr': 1,
            'debug_print': '.FALSE.',
        }

        # Single atom, 2x2 matrices for simplicity
        occupation_numbers = [
            np.array([
                [[1.0, 0.0], [0.0, 0.5]],   # spin up
                [[0.8, 0.0], [0.0, 0.3]],   # spin down
            ])
        ]

        # Call as a static-like method (it doesn't use self beyond method resolution)
        result = ConstrainedPWCalculation.write_oscdft_data(None, parameters, occupation_numbers)

        # Check &OSCDFT section
        assert '&OSCDFT' in result
        assert 'nconstr = 1,' in result
        assert 'debug_print = .FALSE.,' in result
        assert '/' in result

        # Check TARGET_OCCUPATION_NUMBERS section
        assert 'TARGET_OCCUPATION_NUMBERS' in result

        # Check 1-based indexing (atom=1, spin=1/2, orb=1/2)
        lines = result.split('\n')
        target_lines = [l.strip() for l in lines if l.strip() and l.strip()[0].isdigit()]
        assert len(target_lines) == 8  # 1 atom * 2 spins * 2*2 orbitals

        # First line should be: atom=1, spin=1, orb1=1, orb2=1, value=1.000
        first_target = target_lines[0].split()
        assert first_target[0] == '1'  # atom index
        assert first_target[1] == '1'  # spin index
        assert first_target[2] == '1'  # orbital 1
        assert first_target[3] == '1'  # orbital 2
        assert float(first_target[4]) == pytest.approx(1.0)

    def test_multi_atom(self):
        """Test write_oscdft_data with multiple atoms."""
        from lordcapulet.calculations.constrained_pw import ConstrainedPWCalculation

        parameters = {'nconstr': 2}
        dim = 3

        occupation_numbers = [
            np.zeros((2, dim, dim)),  # atom 1
            np.eye(dim).reshape(1, dim, dim).repeat(2, axis=0),  # atom 2: identity for both spins
        ]

        result = ConstrainedPWCalculation.write_oscdft_data(None, parameters, occupation_numbers)

        lines = result.split('\n')
        target_lines = [l.strip() for l in lines if l.strip() and l.strip()[0].isdigit()]

        # 2 atoms * 2 spins * 3*3 = 36 lines
        assert len(target_lines) == 36

        # Last line should have atom index 2
        last = target_lines[-1].split()
        assert last[0] == '2'

    def test_formatting_closure(self):
        """Verify the formatted string ends with a newline-joined block, no trailing newline."""
        from lordcapulet.calculations.constrained_pw import ConstrainedPWCalculation

        parameters = {'nconstr': 1}
        occupation_numbers = [np.zeros((2, 2, 2))]

        result = ConstrainedPWCalculation.write_oscdft_data(None, parameters, occupation_numbers)

        # Should start with &OSCDFT section
        assert result.startswith(' &OSCDFT')
        # Should contain the / closure for namelist
        assert '\n/' in result or result.split('\n')[2] == '/'


class TestPrepareForSubmission:
    """Test ``ConstrainedPWCalculation.prepare_for_submission`` via the generate_calc_job fixture."""

    def test_with_jsonable_data(self, fixture_sandbox, generate_calc_job, generate_inputs_constrained_pw):
        """Test prepare_for_submission with JsonableData target_matrix input."""
        inputs = generate_inputs_constrained_pw(with_jsonable=True)
        calc_info = generate_calc_job(fixture_sandbox, 'lordcapulet.constrained_pw', inputs)

        # oscdft.in must exist in the sandbox
        assert fixture_sandbox.isfile('oscdft.in')

        # Read and verify content
        with fixture_sandbox.open('oscdft.in') as handle:
            content = handle.read()

        assert '&OSCDFT' in content
        assert 'TARGET_OCCUPATION_NUMBERS' in content
        assert 'nconstr = 1,' in content

    def test_with_legacy_dict(self, fixture_sandbox, generate_calc_job, generate_inputs_constrained_pw):
        """Test prepare_for_submission with legacy Dict target_matrix input."""
        inputs = generate_inputs_constrained_pw(with_jsonable=False)
        calc_info = generate_calc_job(fixture_sandbox, 'lordcapulet.constrained_pw', inputs)

        assert fixture_sandbox.isfile('oscdft.in')

        with fixture_sandbox.open('oscdft.in') as handle:
            content = handle.read()

        assert '&OSCDFT' in content
        assert 'TARGET_OCCUPATION_NUMBERS' in content

    def test_oscdft_in_in_retrieve_list(self, fixture_sandbox, generate_calc_job, generate_inputs_constrained_pw):
        """Verify ``oscdft.in`` is added to the retrieve list."""
        inputs = generate_inputs_constrained_pw()
        calc_info = generate_calc_job(fixture_sandbox, 'lordcapulet.constrained_pw', inputs)

        assert 'oscdft.in' in calc_info.retrieve_list

    def test_pw_inputs_preserved(self, fixture_sandbox, generate_calc_job, generate_inputs_constrained_pw):
        """Verify that standard PW input file is still generated alongside oscdft.in."""
        inputs = generate_inputs_constrained_pw()
        calc_info = generate_calc_job(fixture_sandbox, 'lordcapulet.constrained_pw', inputs)

        # aiida.in is the standard QE input
        assert fixture_sandbox.isfile('aiida.in')
        # oscdft.in is the extra file
        assert fixture_sandbox.isfile('oscdft.in')

    def test_file_regression(self, fixture_sandbox, generate_calc_job, generate_inputs_constrained_pw, file_regression):
        """Regression test for the generated oscdft.in content."""
        inputs = generate_inputs_constrained_pw()
        generate_calc_job(fixture_sandbox, 'lordcapulet.constrained_pw', inputs)

        with fixture_sandbox.open('oscdft.in') as handle:
            content = handle.read()

        file_regression.check(content, encoding='utf-8', extension='.in')

    def test_oscdft_flag_injected_without_user_settings(
        self, fixture_sandbox, generate_calc_job, generate_inputs_constrained_pw,
    ):
        """The plugin must add '-oscdft' to pw.x cmdline when the user did
        not supply a ``settings`` input. Without the flag, pw.x silently
        ignores oscdft.in and the constraint never reaches QE."""
        inputs = generate_inputs_constrained_pw()
        assert 'settings' not in inputs

        calc_info = generate_calc_job(
            fixture_sandbox, 'lordcapulet.constrained_pw', inputs,
        )
        assert '-oscdft' in calc_info.codes_info[0].cmdline_params

    def test_oscdft_flag_preserved_when_user_supplies_cmdline(
        self, fixture_sandbox, generate_calc_job, generate_inputs_constrained_pw,
    ):
        """If the user passes their own CMDLINE (e.g. extra MPI/QE flags),
        '-oscdft' is still present in the final cmdline."""
        from aiida.orm import Dict

        inputs = generate_inputs_constrained_pw()
        inputs['settings'] = Dict(dict={
            'CMDLINE': ['-ndiag', '1'],
        })

        calc_info = generate_calc_job(
            fixture_sandbox, 'lordcapulet.constrained_pw', inputs,
        )
        cmdline = calc_info.codes_info[0].cmdline_params
        assert '-oscdft' in cmdline
        assert '-ndiag' in cmdline

    def test_oscdft_flag_not_duplicated_when_user_already_supplied_it(
        self, fixture_sandbox, generate_calc_job, generate_inputs_constrained_pw,
    ):
        """If the user already passed '-oscdft' explicitly, the plugin must
        not double it (pw.x would error on duplicate flags)."""
        from aiida.orm import Dict

        inputs = generate_inputs_constrained_pw()
        inputs['settings'] = Dict(dict={
            'CMDLINE': ['-oscdft'],
        })

        calc_info = generate_calc_job(
            fixture_sandbox, 'lordcapulet.constrained_pw', inputs,
        )
        cmdline = calc_info.codes_info[0].cmdline_params
        assert cmdline.count('-oscdft') == 1

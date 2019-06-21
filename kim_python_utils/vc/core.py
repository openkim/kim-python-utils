################################################################################
#
#  CDDL HEADER START
#
#  The contents of this file are subject to the terms of the Common Development
#  and Distribution License Version 1.0 (the "License").
#
#  You can obtain a copy of the license at
#  http:# www.opensource.org/licenses/CDDL-1.0.  See the License for the
#  specific language governing permissions and limitations under the License.
#
#  When distributing Covered Code, include this CDDL HEADER in each file and
#  include the License file in a prominent location with the name LICENSE.CDDL.
#  If applicable, add the following below this CDDL HEADER, with the fields
#  enclosed by brackets "[]" replaced with your own identifying information:
#
#  Portions Copyright (c) [yyyy] [name of copyright owner]. All rights reserved.
#
#  CDDL HEADER END
#
#  Copyright (c) 2017-2019, Regents of the University of Minnesota.
#  All rights reserved.
#
#  Contributor(s):
#     Ellad B. Tadmor
#
################################################################################
"""
Helper routines for KIM Verification Checks

"""
# Python 2-3 compatible code issues
from __future__ import print_function

import time
import os
import sys
import math
import textwrap

import numpy as np
import jinja2
from ase import Atoms

__version__ = "0.1.0"
__author__ = ["Ellad B. Tadmor"]
__all__ = [
    "KIMVCError",
    "VerificationCheck",
    "setup_and_run_vc",
    "vc_stripall",
    "vc_letter_grade_machine_precision",
]

################################################################################
class KIMVCError(Exception):
    def __init__(self, msg):
        # Call the base class constructor with the parameters it needs
        super(KIMVCError, self).__init__(msg)
        self.msg = msg

    def __str__(self):
        return self.msg


################################################################################
def vc_stripall(string):
    """
    In "string", remove leading and trailing newlines, replace internal newlines
    with spaces, and remove carriage returns.

    Primary purpose is to process docstring at top of VC code and convert to
    vc_description.
    """
    return string.strip().replace("\n", " ").replace("\r", "")


################################################################################
def vc_letter_grade_machine_precision(error, Amax=7.5, Bmax=10.0, Cmax=12.5, Dmax=15.0):
    """
    This function assigns a letter grade in the range A-D and F for fail
    based on the obtained 'error' as compared with the machine precision
    of the computation.
    """
    if not Amax < Bmax < Cmax < Dmax:
        raise KIMVCError("ERROR: This condition not satisfied: Amax<Bmax<Cmax<Dmax")
    eps = np.finfo(float).eps
    if abs(error) < eps:
        letter_grade = "A"
    else:
        score = math.log10(abs(error) / eps)
        if score <= Amax:
            letter_grade = "A"
        elif score <= Bmax:
            letter_grade = "B"
        elif score <= Cmax:
            letter_grade = "C"
        elif score <= Dmax:
            letter_grade = "D"
        else:
            letter_grade = "F"
    # Return with assigned letter grade
    comment = (
        "The letter grade {0:} was assigned because the normalized error in the "
        "computation was {1:.5e} compared with a machine precision of {2:.5e}. The "
        "letter grade was based on 'score=log10(error/eps)', with ranges "
        "A=[0, {3:.1f}], B=({3:.1f}, {4:.1f}], C=({4:.1f}, {5:.1f}], D=({5:.1f}, "
        "{6:.1f}), F>{6:.1f}.  'A' is the best grade, and 'F' indicates failure."
    ).format(letter_grade, error, eps, Amax, Bmax, Cmax, Dmax)
    return letter_grade, comment


################################################################################
def setup_and_run_vc(
    do_vc,
    model,
    vc_name,
    vc_author,
    vc_description,
    vc_category,
    vc_grade_basis,
    vc_files,
    vc_debug,
):
    """
    Construct VerificationCheck object containing all info related to VC,
    and perform the VC by calling the user supplied `do_vc` function.  The VC
    report and property instance re generated.  If `vc_debug` is True, errors
    generate exceptions will full traceback, otherwise exceptions are trapped
    and an error message is reported.
    """

    # Define VC object and do verification check
    vc = VerificationCheck(
        vc_name,
        vc_author,
        vc_description,
        vc_category,
        vc_grade_basis,
        vc_files,
        vc_debug,
    )
    try:
        with vc:
            # Perform verification check and get grade
            try:
                vc_grade, vc_comment = do_vc(model, vc)
                vc.rwrite("Grade: {}".format(vc_grade))
                vc.rwrite("")
                vc.rwrite("Comment: " + vc_comment)
                vc.rwrite("")
            except BaseException as e:
                vc_grade = "N/A"
                vc_comment = "Unable to perform verification check due to an error."
                if vc_debug:
                    raise
                else:
                    vc.rwrite(
                        "\nERROR: Unable to perform verification check.\n\n Message = "
                        + str(e)
                    )
            finally:
                # Pack results in a dictionary and write VC property instance
                results = {
                    "vc_name": vc_name,
                    "vc_description": vc_description,
                    "vc_category": vc_category,
                    "vc_grade_basis": vc_grade_basis,
                    "vc_grade": vc_grade,
                    "vc_comment": vc_comment,
                    "vc_files": vc_files,
                }
                vc.write_results(results)
    except Exception as e:
        if vc_debug:
            raise
        else:
            sys.stderr.write(
                "\nERROR: Unable to initalize verification check.\n\n Message = "
                + str(e)
            )


################################################################################
class VerificationCheck(object):
    """
    Features routines related to outputting VC information and results.
    """

    ############################################################################
    def rwrite(self, string):
        """
        Write 'string' with appended newline to report and echo to stdout
        """
        if not self.report_ready:
            raise KIMVCError(
                "ERROR: Trying to write out string to "
                "report file that is not ready.\n\n"
                '       string =  "{}"\n'.format(string)
            )

        self.report.write(string + "\n")
        sys.stdout.write(string + "\n")

    ############################################################################
    def __init__(
        self,
        vc_name,
        vc_author,
        vc_description,
        vc_category,
        vc_grade_basis,
        vc_files,
        vc_debug,
    ):
        """
        Initialize a VC object.
        """
        # Initialize object variables
        self.output_ready = False  # Output directory not setup
        self.report_ready = False  # Report file not open for writing
        self.vc_name = vc_name
        self.vc_description = vc_description
        self.vc_author = vc_author
        self.vc_category = vc_category
        self.vc_grade_basis = vc_grade_basis
        self.vc_files = vc_files
        self.vc_debug = vc_debug

    ############################################################################
    def __enter__(self):
        """
        Beginning with block:
        Send message to stdout, open report file, print header,
        and return report file object.
        """
        # Send message to stdout that VC calculation is starting
        print()
        print(
            "=== Verification check {} start ({}) ==="
            "".format(self.vc_name, time.strftime("%Y-%m-%d %H:%M:%S"))
        )
        # Create directory 'output' if it does not already exist:
        try:
            if not os.path.exists("output"):
                os.makedirs("output")
            self.output_ready = True
        except (OSError, IOError):
            raise KIMVCError("ERROR: Unable to create directory 'output'.\n")
        # Open report file and write header
        try:
            self.report = open(
                os.path.abspath("output/report.txt"), "w", encoding="utf-8"
            )
            self.report_ready = True
        except (OSError, IOError):
            raise KIMVCError(
                "ERROR: Unable to open verification check report file for writing.\n"
            )
        # Output header to report file
        padside = 5
        c = "!"
        content_line = "VERIFICATION CHECK: " + self.vc_name
        content_width = len(content_line)
        content_line += " " * (content_width - len(content_line))
        width = 2 * padside + 4 + content_width
        self.rwrite(c * width)
        self.rwrite(c * width)
        self.rwrite(c * padside + " " * (content_width + 4) + c * padside)
        self.rwrite(c * padside + "  " + content_line + "  " + c * padside)
        self.rwrite(c * padside + " " * (content_width + 4) + c * padside)
        self.rwrite(c * width)
        self.rwrite(c * width)
        self.rwrite("")
        prefix = "Description: "
        preferredWidth = 80
        wrapper = textwrap.TextWrapper(
            initial_indent=prefix,
            width=preferredWidth,
            subsequent_indent=" " * len(prefix),
        )
        self.rwrite(wrapper.fill(self.vc_description))
        self.rwrite("")
        self.rwrite("Author: {}".format(self.vc_author))

    ############################################################################
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Leaving with block:
        Finishing VC calculation.  Send message to stdout, and close report file.
        """
        # Send message to stdout that VC calculation is starting
        print()
        print(
            "=== Verification check {} end ({}) ==="
            "".format(self.vc_name, time.strftime("%Y-%m-%d %H:%M:%S"))
        )
        # Close report file
        self.report.close()

    ############################################################################
    def write_results(self, results):
        """
        Use jinja2 to Process the template file 'results.edn.tpl' (expected
        in the current working directory) to replace the keys in 'results'.
        Place the new 'results.edn' file in a directory called 'output',
        creating it if it does not already exist.
        """
        if not self.report_ready:
            raise KIMVCError(
                "ERROR: Trying to write out results.edn "
                "but directory 'output' is not ready.\n"
            )

        # Turn lists into strings with double quotes to to be compatible with
        # EDN format
        for key in results:
            if isinstance(results[key], list):
                results[key] = str(results[key]).replace("'", '"')

        template_environment = jinja2.Environment(
            loader=jinja2.FileSystemLoader("/"),
            block_start_string="@[",
            block_end_string="]@",
            variable_start_string="@<",
            variable_end_string=">@",
            comment_start_string="@#",
            comment_end_string="#@",
            undefined=jinja2.StrictUndefined,
        )
        # Template the EDN output
        with open(os.path.abspath("output/results.edn"), "w", encoding="utf-8") as f:
            template = template_environment.get_template(
                os.path.abspath("results.edn.tpl")
            )
            f.write(template.render(**results))

    ############################################################################
    def write_aux_ase_atoms(self, aux_file, atoms, format):
        """
        Write the configuration in the ASE 'atoms' object to an auxiliary
        file 'aux_file' in the output directory in the specified 'format'.
        For supported formats, see https://wiki.fysik.dtu.dk/ase/ase/io/io.html
        """
        if not self.report_ready:
            raise KIMVCError(
                "ERROR: Trying to write out ASE Atoms "
                "aux file but directory 'output' is not ready.\n"
            )

        Atoms.write(atoms, "output/" + aux_file, format=format)

    ############################################################################
    def write_aux_x_y(self, aux_file, x, y):
        """
        Write a file containing the lists x and y:
        x[0] y[0]
        x[1] y[1]
        ...
        If the lists are not the same length, the shorter sets the size.
        """
        if not self.report_ready:
            raise KIMVCError(
                "ERROR: Trying to write out x-y data "
                "aux file but directory 'output' is not ready.\n"
            )

        with open(os.path.abspath("output/" + aux_file), "w", encoding="utf-8") as f:
            for i in range(0, min(len(x), len(y))):
                f.write("{0: 11.8e} {1: 11.8e}\n".format(x[i], y[i]))

    ############################################################################
    def write_aux_string(self, aux_file, string):
        """
        Write a file containing the contents in 'string'
        """
        if not self.report_ready:
            raise KIMVCError(
                "ERROR: Trying to write out string to "
                "aux file but directory 'output' is not ready.\n"
            )

        with open(os.path.abspath("output/" + aux_file), "w", encoding="utf-8") as f:
            f.write("{}\n".format(string))


################################################################################
# If called directly, do nothing
if __name__ == "__main__":
    pass

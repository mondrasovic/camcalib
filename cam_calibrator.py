#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Milan Ondrašovič <milan.ondrasovic@gmail.com>
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pathlib
import sys
import tqdm

import click
import cv2 as cv
import numpy as np


@click.command()
@click.argument("frames_dir_path", type=click.Path(exists=True))
@click.argument("calib_output_dir_path", type=click.Path())
@click.option(
    "-p", "--pattern-size", default=(9, 7), show_default=True, type=(int, int),
    help="Chessboard pattern size."
)
@click.option(
    "-s", "--show-preview", is_flag=True,
    help="Preview frame with detected patterns."
)
@click.option(
    "-d", "--delay", default=1000, show_default=True, type=int,
    help="Time delay between frame previews."
)
@click.option(
    "-v", "--verbose", is_flag=True, help="Enable verbose mode."
)
def main(
        frames_dir_path, calib_output_dir_path, pattern_size, show_preview,
        delay, verbose
):
    """This application is licensed under the MIT license.

    Copyright (c) 2021 Milan Ondrašovič
    """
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    pattern_points = np.zeros(
        (pattern_size[0] * pattern_size[1], 3), np.float32
    )
    pattern_points[:, :2] =\
        np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    obj_points = []  # 3D points in real world space.
    img_points = []  # 2D points in image plane.
    
    frames_iter = pathlib.Path(frames_dir_path).iterdir()
    
    if verbose:
        frames_pbar = tqdm.tqdm(frames_iter, desc="processing frame")
    else:
        frames_pbar = map(lambda x: x, frames_iter)
    
    for frame_file in frames_pbar:
        frame = cv.imread(str(frame_file))
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        ret, corners = cv.findChessboardCorners(frame_gray, pattern_size)
        if not ret:
            continue
    
        obj_points.append(pattern_points)
    
        corners_refined = cv.cornerSubPix(
            frame_gray, corners, (11, 11), (-1, -1), criteria
        )
        img_points.append(corners_refined)
    
        if show_preview:
            frame = cv.drawChessboardCorners(
                frame, pattern_size, corners_refined, ret
            )
            cv.imshow("Calibration frame preview", frame)
            if cv.waitKey(delay) & 0xff == ord('q'):
                break

    cv.destroyAllWindows()

    _, camera_matrix, dist_coefs, _, _ = cv.calibrateCamera(
        obj_points, img_points, pattern_size, None, None
    )
    
    output_dir = pathlib.Path(calib_output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(str(output_dir / "camera_matrix.npy"), camera_matrix)
    np.save(str(output_dir / "dist_coefficients.npy"), dist_coefs)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

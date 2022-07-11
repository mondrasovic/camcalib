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

import sys

import click
import cv2 as cv
import numpy as np


def draw_world_axes(img, rvec, tvec, K, dist_coefs=(0, 0, 0, 0), mag=3):
    def _draw_line(pt1, pt2, color):
        cv.arrowedLine(
            img, pt1, pt2, color=color, thickness=3, line_type=cv.LINE_AA,
            tipLength=0.2
        )

    points = np.float32(
        ((mag, 0, 0), (0, mag, 0), (0, 0, mag), (0, 0, 0))
    )
    points_projected, _ = cv.projectPoints(points, rvec, tvec, K, dist_coefs)
    origin = tuple(points_projected[3].ravel())

    _draw_line(origin, tuple(points_projected[0].ravel()), (0, 0, 255))
    _draw_line(origin, tuple(points_projected[1].ravel()), (0, 255, 0))
    _draw_line(origin, tuple(points_projected[2].ravel()), (255, 0, 0))

    return img


def draw_world_plane(
        img, rvec, tvec, K, dist_coefs=(0, 0, 0, 0), mag=10, alpha=0.7
):
    points = np.float32(
        ((-mag, -mag, 0), (mag, -mag, 0),
         (mag, mag, 0), (-mag, mag, 0))
    )
    points_projected, _ = cv.projectPoints(points, rvec, tvec, K, dist_coefs)
    points_projected = points_projected.round().astype(np.int32)

    img_overlay = np.ones_like(img, dtype=np.float32)
    cv.fillPoly(img_overlay, [points_projected], (alpha, alpha, alpha))
    img = (img * img_overlay).round().astype(np.uint8)

    return img


@click.command()
@click.argument("cam_matrix_file_path", type=click.Path(exists=True))
@click.option(
    "--dist-coefs-file-path", type=click.Path(exists=True),
    help="File pdath containing distortion coefficients in NumPy format"
)
@click.option(
    "-p", "--pattern-size", default=(9, 7), show_default=True, type=(int, int),
    help="Chessboard pattern size."
)
@click.option(
    "-c", "--show-cam-pos", is_flag=True,
    help="Show camera position in 3D world coordinates."
)
def main(
        cam_matrix_file_path, dist_coefs_file_path, pattern_size, show_cam_pos
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

    cam_matrix = np.load(cam_matrix_file_path)
    dist_coefs = np.load(dist_coefs_file_path)
    
    capture = cv.VideoCapture(0, cv.CAP_DSHOW)
    
    quit_key = 'q'
    
    while True:
        read_status, frame = capture.read()
        if not read_status:
            print(
                "error: unable to read next frame, terminating...",
                file=sys.stderr
            )
            break
        
        frame = cv.flip(frame, 1)
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(frame_gray, pattern_size)
        if ret:
            corners_refined = cv.cornerSubPix(
                frame_gray, corners, (11, 11), (-1, -1), criteria
            )
            ret, rvec, tvec = cv.solvePnP(
                pattern_points, corners_refined, cam_matrix, dist_coefs
            )
            frame = draw_world_plane(frame, rvec, tvec, cam_matrix, dist_coefs)
            frame = draw_world_axes(frame, rvec, tvec, cam_matrix, dist_coefs)
            
            if show_cam_pos:
                rmat, _ = cv.Rodrigues(rvec)
                cam_pos = np.squeeze(rmat.T @ (-tvec))
                coords = ','.join(f"{c:.2f}" for c in cam_pos)
                cam_pos_text = f"cam. pos.: [{coords}]"
                cv.putText(
                    frame, cam_pos_text, (20, 30), cv.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255, 255, 255), thickness=1,
                    lineType=cv.LINE_AA
                )
        
        cv.imshow(f"Webcam feed (press \'{quit_key}\' to quit)", frame)
        if cv.waitKey(1) & 0xff == ord(quit_key):
            break
    
    capture.release()
    cv.destroyAllWindows()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

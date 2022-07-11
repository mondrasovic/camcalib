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
import time

import click
import cv2 as cv
import numpy as np


def draw_countdown_clock(img, progress, radius=0.5, origin=None, alpha=0.5):
    radius = int(round((min(img.shape[:2]) / 2) * radius))
    
    if origin:
        center = origin
    else:
        cx, cy = img.shape[1] // 2, img.shape[0] // 2
        center = (cx, cy)
    
    angle = np.pi * (2 * progress - 0.5)
    
    fill_mask = np.ones_like(img, dtype=np.float32)
    cv.ellipse(
        fill_mask, center, (radius, radius), 0,
        270, np.degrees(angle), (alpha, alpha, alpha), thickness=-1,
        lineType=cv.LINE_AA
    )
    img = (img * fill_mask).round().astype(np.uint8)
    
    white = (255, 255, 255)
    cv.circle(img, center, radius, white, thickness=1, lineType=cv.LINE_AA)
    cv.circle(img, center, 3, white, thickness=-1, lineType=cv.LINE_AA)
    
    coords = np.asarray((np.cos(angle), np.sin(angle)))
    coords = ((coords * radius) + center).round().astype(np.int)
    cv.line(img, center, tuple(coords), white, thickness=1, lineType=cv.LINE_AA)
    
    return img


def draw_progress_bar(img, progress):
    pass


@click.command()
@click.argument("output_dir_path", type=click.Path())
@click.option(
    "-n", "--n-frames", type=int, default=5, show_default=True,
    help="Number of frames to grab."
)
@click.option(
    "-s", "--n-seconds", type=int, default=3, show_default=True,
    help="Number of seconds between frames."
)
@click.option(
    "-p", "--show-progress-bar", is_flag=True, help="Show progress bar."
)
@click.option(
    "-c", "--show-countdown", is_flag=True,
    help="Show graphical countdown clock."
)
@click.option(
    "-v", "--verbose", is_flag=True,
    help="Enable verbose mode."
)
def main(
        output_dir_path, n_frames, n_seconds, show_progress_bar, show_countdown,
        verbose
):
    """This application is licensed under the MIT license.
    
    Copyright (c) 2021 Milan Ondrašovič
    
    A utility to grab a desired amount of frames from a webcam feed. The purpose
    may be to obtain various positions of artificual markers for camera
    calibration. The user needs to have enough time to induce a change in the
    scene, thus a delay between grabbed frames is desired.
    """
    capture = cv.VideoCapture(0, cv.CAP_DSHOW)
    quit_key = 'q'
    prev_time, frame_count = time.time(), 0
    
    output_dir = pathlib.Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    while frame_count < n_frames:
        read_status, frame = capture.read()
        if not read_status:
            print(
                "error: unable to read next frame, terminating...",
                file=sys.stderr
            )
            break
        
        curr_time = time.time()
        elapsed_time = curr_time - prev_time
        
        if elapsed_time >= n_seconds:
            frame_count += 1
            prev_time = curr_time
            
            file_name = f"frame_{frame_count:04d}.jpg"
            file_path = str(output_dir / file_name)
            write_status = cv.imwrite(file_path, frame)
            
            if verbose:
                if write_status:
                    file, status = sys.stdout, "success"
                else:
                    file, status = sys.stderr, "failure"
                print(f"saving frame {file_path}: {status}", file=file)
        
        if show_countdown:
            progress = min(elapsed_time, n_seconds) / n_seconds
            frame = draw_countdown_clock(frame, progress)
        
        cv.imshow(f"Webcam feed (press \'{quit_key}\' to quit)", frame)
        if cv.waitKey(1) & 0xff == ord(quit_key):
            break
    
    capture.release()
    cv.destroyAllWindows()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

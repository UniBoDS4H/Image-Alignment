package com.ds4h.model.util;

import org.apache.commons.io.FilenameUtils;

import java.io.File;
import java.util.Objects;

//TODO: Add doc
public class CheckImage {

    private static final double UPPER_LIMIT = 4; // Max dimension of an image in Java is 4GB

    private static final String EXTENSIONS ="jpg, jpeg, png, gif, bmp, tiff, tif, webp, svg, heif," +
            " heic, raw, arw, cr2, nef, orf, " +
            "rw2, dng, psd, pcx," +
            " ppm, pgm, pbm, dds, hdr, exr, pfm, icns, xbm, xpm, pict, jp2, jpx, pcd";
    private CheckImage(){}
    public static boolean checkImage(final File file) throws IllegalArgumentException{
        if(Objects.nonNull(file) && file.isFile()) {
            final String fileExtension = FilenameUtils.getExtension(file.getName());
            if(CheckImage.EXTENSIONS.contains(fileExtension.toLowerCase())){
                return CheckImage.checkSize(file);
            }
        }
        return false;
    }

    public static boolean isTiff(final File file){
        final String fileExtension = FilenameUtils.getExtension(Objects.requireNonNull(file.getName()));
        return !fileExtension.isEmpty() && (fileExtension.contains("tif")
                || fileExtension.contains("tiff"));
    }

    private static boolean checkSize(final File image) throws IllegalArgumentException{
        final double length = CheckImage.getFileSizeGigaBytes(image);
        if(length < UPPER_LIMIT){
            return true;
        }
        throw new IllegalArgumentException("The image : " + image.getName() + " is too big. Java can not support the image's size.");
    }

    private static double getFileSizeGigaBytes(final File file) {
        final double div = Math.pow(1024, 3);
        return (double) file.length() / (div);
    }
}

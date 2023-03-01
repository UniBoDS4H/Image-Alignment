package com.ds4h.model.alignment.manual;

import com.ds4h.model.alignedImage.AlignedImage;
import com.ds4h.model.alignment.AlignmentAlgorithm;
import com.ds4h.model.imagePoints.ImagePoints;
import ij.ImagePlus;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.Optional;

/**
 * This class is used for the manual alignment using the Affine technique
 */
public class AffineAlignment extends AlignmentAlgorithm {
    public static final int REQUIRED_POINTS = 3;
    public AffineAlignment(){
        super();
    }

    /**
     * Manual alignment using the Affine alignment
     * @param source : the source image used as reference
     * @param  target : the target to align
     * @throws IllegalArgumentException : in case the number of corners is not correct
     * @return : the list of all the images aligned to the source
     */
    @Override
    protected Optional<AlignedImage> align(final ImagePoints source, final ImagePoints target) throws IllegalArgumentException{
        try {
            if(source.numberOfPoints() == REQUIRED_POINTS && target.numberOfPoints() == REQUIRED_POINTS) {
                final MatOfPoint2f referencePoint = source.getMatOfPoint();
                final MatOfPoint2f targetPoint = target.getMatOfPoint();
                final Mat H = Imgproc.getAffineTransform(targetPoint, referencePoint);
                final Mat warpedMat = new Mat();
                Imgproc.warpAffine(target.getMatImage(), warpedMat, H, source.getMatImage().size(), Imgproc.INTER_LINEAR, 0, new Scalar(0, 0, 0));
                final Optional<ImagePlus> finalImage = this.convertToImage(target.getFile(), warpedMat);
                return finalImage.map(imagePlus -> new AlignedImage(warpedMat, H, imagePlus));
            }else{
                throw new IllegalArgumentException("The number of points inside the source image or inside the target image is not correct.\n" +
                        "In order to use the Affine alignment you must use: " + AffineAlignment.REQUIRED_POINTS + " points.");
            }
        }catch (Exception ex){
            throw ex;
        }
    }

    @Override
    public int neededPoints(){
        return AffineAlignment.REQUIRED_POINTS;
    }
}

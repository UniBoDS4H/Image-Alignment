package com.ds4h.model.alignment.manual;

import com.ds4h.model.alignedImage.AlignedImage;
import com.ds4h.model.alignment.AlignmentAlgorithm;
import com.ds4h.model.imagePoints.ImagePoints;
import ij.ImagePlus;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;

import java.awt.*;
import java.util.Arrays;
import java.util.Optional;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * This class is used for the manual alignment using the Translative technique
 */
public class TranslationalAlignment extends AlignmentAlgorithm {

    public static final int LOWER_BOUND = 1;

    public TranslationalAlignment(){
        super();
    }

    /**
     * Manual alignment using the translative alignment
     * @param targetImage : the source image used as reference
     * @param  imagePoints : the target to align
     * @throws IllegalArgumentException : in case the number of corners is not correct
     * @return : the list of all the images aligned to the source
     */
    @Override
    protected Optional<AlignedImage> align(final ImagePoints targetImage, final ImagePoints imagePoints) throws IllegalArgumentException{
        try {
            MatOfPoint2f tarO = new MatOfPoint2f();
            MatOfPoint2f srcO = new MatOfPoint2f();
            ransac(imagePoints.getMatOfPoint(), targetImage.getMatOfPoint(), 100,10,srcO, tarO);
            //Arrays.stream(targetImage.getPoints()).forEach(targetImage::removePoint);
            //Arrays.stream(imagePoints.getPoints()).forEach(imagePoints::removePoint);
            System.out.println(srcO.toList());
            System.out.println(tarO.toList());
            srcO.toList().forEach(imagePoints::addPoint);
            tarO.toList().forEach(targetImage::addPoint);

            if(targetImage.numberOfPoints() >= LOWER_BOUND && imagePoints.numberOfPoints() >= LOWER_BOUND) {
                final Mat targetMat = targetImage.getMatImage();
                final Mat imageToShiftMat = imagePoints.getMatImage();

                final Point[] srcArray = targetImage.getMatOfPoint().toArray();
                final Point[] dstArray = imagePoints.getMatOfPoint().toArray();
                if(srcArray.length == dstArray.length) {
                    final Mat alignedImage = new Mat();
                    final Mat translationMatrix = this.getTransformationMatrix(imagePoints, targetImage);//super.traslationMatrix(imagePoints);
                    Imgproc.warpPerspective(imageToShiftMat, alignedImage, translationMatrix, targetMat.size());
                    final Optional<ImagePlus> finalImage = this.convertToImage(imagePoints.getFile(), alignedImage);
                    return finalImage.map(imagePlus -> new AlignedImage(alignedImage, translationMatrix, imagePlus));
                }else{
                    throw new IllegalArgumentException("The number of corner inside the source image is different from the number of points" +
                            "inside the target image.");
                }
            }else{
                throw new IllegalArgumentException("The number of points inside the source image or inside the target image is not correct.\n" +
                        "In order to use the Translation alignment you must at least: " + TranslationalAlignment.LOWER_BOUND + " points.");
            }
        }catch (Exception ex){
            throw ex;
        }
    }
    public void ransac(MatOfPoint2f src, MatOfPoint2f dst, int num, double threshold, MatOfPoint2f inliersSrc, MatOfPoint2f inliersDst) {

        int numPoints = src.rows();


        // Initialize variables
        Random rand = new Random();
        int maxInliers = 0;
        MatOfPoint2f bestInliersSrc = new MatOfPoint2f();
        MatOfPoint2f bestInliersDst = new MatOfPoint2f();

        // Run RANSAC for a fixed number of iterations
        for (int iter = 0; iter < num; iter++) {
            System.out.println("numPoints->"+ numPoints);
            // Select a random sample of correspondences
            int[] indices = rand.ints(numPoints, 0, numPoints).distinct().limit(2).toArray();
            Arrays.stream(indices).forEach(System.out::println);
            System.out.println();
            src.toList().forEach(System.out::println);
            System.out.println();


            // Compute the homography using the random sample
            MatOfPoint2f srcSample = new MatOfPoint2f();
            MatOfPoint2f dstSample = new MatOfPoint2f();
            for (int i = 0; i < 2; i++) {
                srcSample.push_back(src.row(indices[i]));
                dstSample.push_back(dst.row(indices[i]));
            }
            Mat H = getMat(srcSample, dstSample);

            // Compute the inliers using the homography and the threshold
            MatOfPoint2f inliersSrcTmp = new MatOfPoint2f();
            MatOfPoint2f inliersDstTmp = new MatOfPoint2f();
            for (int i = 0; i < numPoints; i++) {
                Mat srcPoint = src.row(i);
                Mat dstPoint = dst.row(i);
                double[] srcArray = srcPoint.get(0, 0);
                double[] dstArray = dstPoint.get(0, 0);
                MatOfPoint2f srcMat = new MatOfPoint2f(new Point(srcArray));
                MatOfPoint2f dstMat = new MatOfPoint2f(new Point(dstArray));
                MatOfPoint2f dstTransformed = new MatOfPoint2f();
                Core.perspectiveTransform(srcMat, dstTransformed, H);
                double distance = Core.norm(dstMat, dstTransformed);
                if (distance < threshold) {
                    inliersSrcTmp.push_back(srcMat);
                    inliersDstTmp.push_back(dstMat);
                }
            }

            // Update the best set of inliers
            int numInliers = (int) inliersSrcTmp.total();
            if (numInliers > maxInliers) {
                maxInliers = numInliers;
                bestInliersSrc = inliersSrcTmp;
                bestInliersDst = inliersDstTmp;
            }
        }
        System.out.println("best"+ bestInliersSrc.toList());

        // Return the best set of inliers
        inliersSrc.fromList(bestInliersSrc.toList());
        inliersDst.fromList(bestInliersDst.toList());
    }
    public Mat getMat(final MatOfPoint2f imageToAlign, final MatOfPoint2f targetImage){
        final Point translation = minimumLeastSquare(imageToAlign.toArray(), targetImage.toArray());
        final Mat translationMatrix = Mat.eye(3, 3, CvType.CV_32FC1);
        translationMatrix.put(0, 2, translation.x);
        translationMatrix.put(1, 2, translation.y);
        //super.addMatrix(imageToAlign, translationMatrix);
        return translationMatrix;
    }


    @Override
    public Mat getTransformationMatrix(final ImagePoints imageToAlign, final ImagePoints targetImage){
        final Point translation = minimumLeastSquare(imageToAlign.getPoints(), targetImage.getPoints());
        final Mat translationMatrix = Mat.eye(3, 3, CvType.CV_32FC1);
        translationMatrix.put(0, 2, translation.x);
        translationMatrix.put(1, 2, translation.y);
        super.addMatrix(imageToAlign, translationMatrix);
        return translationMatrix;
    }

    private static Point minimumLeastSquare(final Point[] srcArray, final Point[] dstArray){
        final double[] deltaX = new double[srcArray.length];
        final double[] deltaY = new double[srcArray.length];


        IntStream.range(0, srcArray.length).parallel().forEach(i -> {
            deltaX[i] = dstArray[i].x - srcArray[i].x;
            deltaY[i] = dstArray[i].y - srcArray[i].y;
        });

        final double meanDeltaX = Core.mean(new MatOfDouble(deltaX)).val[0];
        final double meanDeltaY = Core.mean(new MatOfDouble(deltaY)).val[0];
        return new Point(meanDeltaX, meanDeltaY);
    }

    @Override
    public void transform(final Mat source, final Mat destination, final Mat H){
        Core.perspectiveTransform(source, destination, H);
    }

}

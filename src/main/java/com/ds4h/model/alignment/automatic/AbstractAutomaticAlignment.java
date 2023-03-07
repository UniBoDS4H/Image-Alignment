package com.ds4h.model.alignment.automatic;

import com.ds4h.model.alignedImage.AlignedImage;
import com.ds4h.model.alignment.ManualAlgorithm;
import com.ds4h.model.alignment.automatic.pointDetector.PointDetector;
import com.ds4h.model.alignment.preprocessImage.TargetImagePreprocessing;
import com.ds4h.model.imagePoints.ImagePoints;
import com.ds4h.model.pointManager.PointManager;
import com.ds4h.model.util.ImagingConversion;
import com.ds4h.model.util.Pair;
import ij.ImagePlus;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;

public abstract class AbstractAutomaticAlignment implements Runnable{

    private final PointDetector pointDetector;
    private ImagePoints targetImage;
    private final List<ImagePoints> imagesToAlign;
    private final List<AlignedImage> alignedImages;
    private final TargetImagePreprocessing targetImagePreprocessing;
    private Thread alignmentThread;
    protected AbstractAutomaticAlignment(final PointDetector pointDetector){
        this.targetImage = null;
        this.pointDetector = pointDetector;
        this.targetImagePreprocessing = new TargetImagePreprocessing();
        this.alignmentThread = new Thread(this);
        this.imagesToAlign = new LinkedList<>();
        this.alignedImages = new CopyOnWriteArrayList<>();
    }

    protected PointDetector pointDetector(){
        return this.pointDetector;
    }

    public abstract void detectPoint(final Mat targetMat, final ImagePoints imagePoints);
    public abstract void mergePoint(final ImagePoints targetImage, final ImagePoints imagePoints);

    public Mat getTransformationMatrix(final ImagePoints targetImage, final ImagePoints imagePoints){
        final MatOfPoint2f points1_ = new MatOfPoint2f();
        final MatOfPoint2f points2_ = new MatOfPoint2f();
        points1_.fromList(this.pointDetector().getPoints1());
        points2_.fromList(this.pointDetector().getPoints2());
        return Calib3d.findHomography(points1_, points2_, Calib3d.RANSAC, 5);
    }

    public void transform(final Mat source, final Mat destination, final Mat H){
        Core.perspectiveTransform(source, destination, H);
    }

    public Optional<AlignedImage> align(final MatOfPoint2f targetPoints, final ImagePoints imagePoints, final Size targetSize){
        final Mat imagePointMat = imagePoints.getGrayScaleMat();
        final Mat H = Calib3d.findHomography(imagePoints.getMatOfPoint(), targetPoints, Calib3d.RANSAC, 5);
        return this.warpMatrix(imagePointMat, H, targetSize, imagePoints);
    }
    private Optional<AlignedImage> warpMatrix(final Mat source, final Mat H, final Size size, final ImagePoints alignedFile){
        final Mat alignedImage1 = new Mat();
        Imgproc.warpPerspective(source, alignedImage1, H, size);
        final Optional<ImagePlus> finalImage = this.convertToImage(alignedFile.getName(), alignedImage1);
        return finalImage.map(imagePlus -> new AlignedImage(alignedImage1, H, imagePlus));
    }
    private Optional<ImagePlus> convertToImage(final String name, final Mat matrix){
        return ImagingConversion.fromMatToImagePlus(matrix, name);
    }

    public void alignImages(final PointManager pointManager){
        if(Objects.nonNull(pointManager) && pointManager.getSourceImage().isPresent()) {
            if(!this.isAlive()) {
                this.targetImage = pointManager.getSourceImage().get();
                this.alignedImages.clear();
                this.imagesToAlign.clear();
                try {
                    this.imagesToAlign.addAll(pointManager.getImagesToAlign());
                    this.alignmentThread.start();
                } catch (final Exception ex) {
                    throw new IllegalArgumentException("Error: " + ex.getMessage());
                }
            }
        }else{
            throw new IllegalArgumentException("In order to do the alignment It is necessary to have a target," +
                    " please pick a target image.");
        }
    }

    public boolean isAlive(){
        return this.alignmentThread.isAlive();
    }

    public List<AlignedImage> alignedImages(){
        return this.alignedImages;
    }

    @Override
    public void run(){
        try {
            final Pair<ImagePoints, Map<ImagePoints, MatOfPoint2f>> pair = this.targetImagePreprocessing.processAutomaticImage(targetImage, this.imagesToAlign, this);
            final ImagePoints result = pair.getFirst();
            this.alignedImages.add(new AlignedImage(result.getMatImage(), result.getImage()));
            pair.getSecond().entrySet().parallelStream()
                    .forEach(img -> this.align(img.getValue(), img.getKey(), result.getMatImage().size())
                            .ifPresent(this.alignedImages::add));
            this.alignmentThread = new Thread(this);
        }catch (Exception exception){
            this.alignmentThread = new Thread(this);
        }
    }

}
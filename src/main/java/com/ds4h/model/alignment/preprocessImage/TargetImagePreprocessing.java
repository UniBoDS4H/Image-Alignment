package com.ds4h.model.alignment.preprocessImage;

import com.ds4h.model.alignment.AlignmentAlgorithm;
import com.ds4h.model.alignment.automatic.AbstractAutomaticAlignment;
import com.ds4h.model.imagePoints.ImagePoints;
import com.ds4h.model.util.ImagingConversion;
import com.ds4h.model.util.Pair;
import com.ds4h.model.util.directoryManager.directoryCreator.DirectoryCreator;
import com.ds4h.model.util.saveProject.SaveImages;
import ij.ImagePlus;
import org.opencv.core.*;
import org.opencv.core.Point;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;

public class TargetImagePreprocessing {

    //Image -> Punti del target
    public final Map<ImagePoints, MatOfPoint2f> map = new HashMap<>();
    public Mat lastTargetImage = null;
    private final static String DIRECTORY_NAME = "DS4H_processedTarget";
    private final static String TMP_PATH = System.getProperty("java.io.tmpdir");
    public TargetImagePreprocessing(){}
    static public ImagePoints process(final ImagePoints targetImage, final List<ImagePoints> imagesToAlign, final AlignmentAlgorithm algorithm) throws IllegalArgumentException{
        Pair<Mat, MatOfPoint2f> target = new Pair<>(targetImage.getMatImage(),targetImage.getMatOfPoint());

        for (final ImagePoints image : imagesToAlign) {
            target = TargetImagePreprocessing.singleProcess(target.getFirst(), target.getSecond(), targetImage, image, algorithm);
        }
        final String directoryName = DirectoryCreator.createTemporaryDirectory(TargetImagePreprocessing.DIRECTORY_NAME);
        final Optional<ImagePlus> imagePlus = ImagingConversion.fromMatToImagePlus(target.getFirst(),targetImage.getFile().getName());
        if(imagePlus.isPresent()){
            imagePlus.get().setTitle(targetImage.getFile().getName());
            SaveImages.save(imagePlus.get(), TargetImagePreprocessing.TMP_PATH + "/" + directoryName);
            final ImagePoints result = new ImagePoints(new File(TargetImagePreprocessing.TMP_PATH + "/" +directoryName+"/" + targetImage.getFile().getName()));
            target.getSecond().toList().forEach(result::addPoint);
            return result;
        }else{
            throw new IllegalArgumentException("the file doesn't exist");
        }

    }

    private static Pair<Mat,MatOfPoint2f> singleProcess(final Mat targetMat, final MatOfPoint2f targetPoints, final ImagePoints targetImage,  final ImagePoints imagePoints, final AlignmentAlgorithm algorithm) {
        try {
            final Mat imageToShiftMat = imagePoints.getMatImage();

            final Point[] srcArray = targetPoints.toArray();
            final Point[] dstArray = imagePoints.getMatOfPoint().toArray();
            final Mat translationMatrix = algorithm.getTransformationMatrix(imagePoints, targetImage);
            final int h1 = targetMat.rows();
            final int w1 = targetMat.cols();
            final int h2 = imageToShiftMat.rows();
            final int w2 = imageToShiftMat.cols();

            final MatOfPoint2f pts1 = new MatOfPoint2f(new Point(0, 0), new Point(0, h1), new Point(w1, h1), new Point(w1, 0));
            final MatOfPoint2f pts2 = new MatOfPoint2f(new Point(0, 0), new Point(0, h2), new Point(w2, h2), new Point(w2, 0));
            final MatOfPoint2f pts2_ = new MatOfPoint2f();

            algorithm.transform(pts2, pts2_, translationMatrix);

            final MatOfPoint2f pts = new MatOfPoint2f();
            Core.hconcat(Arrays.asList(pts1, pts2_), pts);
            if(!pts.toList().isEmpty()) {
                final Point pts_min = new Point(pts.toList().stream().map(p -> p.x).min(Double::compareTo).get(), pts.toList().stream().map(p -> p.y).min(Double::compareTo).get());
                final Point pts_max = new Point(pts.toList().stream().map(p -> p.x).max(Double::compareTo).get(), pts.toList().stream().map(p -> p.y).max(Double::compareTo).get());


                final int xmin = (int) Math.floor(pts_min.x - 0.5);
                final int ymin = (int) Math.floor(pts_min.y - 0.5);
                final int xmax = (int) Math.ceil(pts_max.x + 0.5);
                final int ymax = (int) Math.ceil(pts_max.y + 0.5);
                final double[] t = {-xmin, -ymin};

                final Size s = new Size(xmax - xmin, ymax - ymin);
                final Mat alignedImage = Mat.zeros(s, imageToShiftMat.type());
                targetMat.copyTo(alignedImage.submat(new Rect((int) t[0], (int) t[1], w1, h1)));
                final MatOfPoint2f points = new MatOfPoint2f();
                points.fromList(targetPoints.toList().stream().map(p -> new Point(p.x + t[0], p.y + t[1])).collect(Collectors.toList()));
                return new Pair<>(alignedImage, points);
            }else{
                throw new IllegalArgumentException("Please check your images.");
            }
        }catch (Exception ex){
            throw ex;
        }
    }

    public Pair<Mat, Map<ImagePoints, MatOfPoint2f>> automaticProcess(final Mat targetMat, final MatOfPoint2f targetPoints, final ImagePoints targetImage,
                                                                      final List<ImagePoints> imagesPoints,
                                                                      final AbstractAutomaticAlignment automaticAlignment){
        for(final ImagePoints i : imagesPoints) {
            final ImagePoints imagePoints = new ImagePoints(i.getFile());
            final ImagePoints newTarget = new ImagePoints(targetImage.getFile());
            if (lastTargetImage != null) {
                automaticAlignment.detectPoint(lastTargetImage, imagePoints);
            } else {
                this.lastTargetImage = targetMat;
                automaticAlignment.detectPoint(newTarget.getMatImage(), imagePoints);
            }
            automaticAlignment.mergePoint(newTarget, imagePoints);
            final Mat translationMatrix = automaticAlignment.getTransformationMatrix(newTarget, imagePoints);
            System.out.println(translationMatrix);
            final int h1 = lastTargetImage.rows();
            final int w1 = lastTargetImage.cols();
            final int h2 = imagePoints.getMatImage().rows();
            final int w2 = imagePoints.getMatImage().cols();

            final MatOfPoint2f pts1 = new MatOfPoint2f(new Point(0, 0), new Point(0, h1), new Point(w1, h1), new Point(w1, 0));
            final MatOfPoint2f pts2 = new MatOfPoint2f(new Point(0, 0), new Point(0, h2), new Point(w2, h2), new Point(w2, 0));
            final MatOfPoint2f pts2_ = new MatOfPoint2f();

            automaticAlignment.transform(pts2, pts2_, translationMatrix);

            final MatOfPoint2f pts = new MatOfPoint2f();
            Core.hconcat(Arrays.asList(pts1, pts2_), pts);
            if (!pts.toList().isEmpty()) {
                final Point pts_min = new Point(pts.toList().stream().map(p -> p.x).min(Double::compareTo).get(), pts.toList().stream().map(p -> p.y).min(Double::compareTo).get());
                final Point pts_max = new Point(pts.toList().stream().map(p -> p.x).max(Double::compareTo).get(), pts.toList().stream().map(p -> p.y).max(Double::compareTo).get());


                final int xmin = (int) Math.floor(pts_min.x - 0.5);
                final int ymin = (int) Math.floor(pts_min.y - 0.5);
                final int xmax = (int) Math.ceil(pts_max.x + 0.5);
                final int ymax = (int) Math.ceil(pts_max.y + 0.5);
                final double[] t = {-xmin, -ymin};

                final Size size = new Size(xmax - xmin, ymax - ymin);
                final Mat aligneImage = Mat.zeros(size, targetImage.getMatImage().type());
                this.lastTargetImage.copyTo(aligneImage.submat(new Rect((int) t[0], (int) t[1], w1, h1)));
                this.lastTargetImage = aligneImage;

                final MatOfPoint2f points = new MatOfPoint2f();
                System.out.println("INSIDE PREPROCESS: " + newTarget.getMatOfPoint());
                points.fromList(newTarget.getMatOfPoint().toList().stream().map(p -> new Point(p.x + t[0], p.y + t[1])).collect(Collectors.toList()));
                this.map.entrySet().parallelStream().forEach(entry -> {
                    entry.getValue().fromList(entry.getValue().toList().stream().map(p -> new Point(p.x + t[0], p.y + t[1])).collect(Collectors.toList()));
                });
                this.map.put(imagePoints, points);
            }
        }
        return new Pair<>(this.lastTargetImage, this.map);
    }
}

package com.ds4h.model.imageCorners;

import com.ds4h.model.util.ImagingConversion;
import ij.ImagePlus;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

public class ImageCorners {
    private final File image;
    private final List<Point> corners;

    public ImageCorners(File image){
        this.image = image;
        this.corners = new ArrayList<>();
    }
    public BufferedImage getBufferedImage(){
        Mat mat = this.getMatImage();
        return new BufferedImage(mat.width(), mat.height(), BufferedImage.TYPE_3BYTE_BGR);
    }

    public String getPath(){
        return this.image.getPath();
    }

    public Point[] getCorners(){
        return this.corners.toArray(new Point[0]);
    }
    public int getIndexOfCorner(Point corner){
        return this.corners.indexOf(corner)+1;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ImageCorners that = (ImageCorners) o;
        return Objects.equals(image, that.image) && Objects.equals(corners, that.corners);
    }

    @Override
    public int hashCode() {
        return Objects.hash(image, corners);
    }

    public void addCorner(Point corner){
        if(!this.corners.contains(corner)){
            this.corners.add(corner);
        }
    }

    public File getFile(){
        return this.image;
    }
    public void removeCorner(Point corner){
        if(!this.corners.remove(corner)){
            throw new IllegalArgumentException("given corner was not found");
        }
    }

    public MatOfPoint2f getMatOfPoint(){
        MatOfPoint2f mat = new MatOfPoint2f();
        mat.fromList(this.corners);
        return mat;
    }

    public Mat getMatImage(){
        return Imgcodecs.imread(this.image.getPath());
    }
    public void moveCorner(File image, Point corner, Point newCorner){

    }
}

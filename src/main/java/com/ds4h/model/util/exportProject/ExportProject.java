package com.ds4h.model.util.exportProject;

import com.ds4h.model.imageCorners.ImageCorners;
import com.ds4h.model.util.directoryCreator.DirectoryCreator;
import com.ds4h.model.util.saveProject.SaveImages;
import org.json.JSONArray;
import org.json.JSONObject;
import org.opencv.core.Point;

import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

//TODO : ADD COLORS INFORMATIONS FOR POINTS
//TODO : ADD INFORMATIONS ABOUT TARGET

public class ExportProject {
    public final static String FILE_KEY = "FILE_NAME";
    public final static String POINTS_KEY = "POINTS";
    private final static String PROJECT_NAME = "ds4h_project.json";
    private final static String PROJECT_FOLDER = "DS4H_Project";
    private ExportProject(){

    }

    //TODO: add better doc
    /**
     * Export the entire project
     * @param images the images with their points.
     * @param path the path where store the project
     * @throws IOException error in the saving s
     */
    public static void exportProject(final List<ImageCorners> images,final String path) throws IOException {

        final JSONArray imageList = ExportProject.createJSON(images);
        final String directory = DirectoryCreator.createDirectory(path, PROJECT_FOLDER);
        if(!directory.isEmpty()){
            SaveImages.save(images.stream().map(ImageCorners::getImage).collect(Collectors.toList()), path+"/"+directory);
            ExportProject.exportJSON(imageList, path+"/"+directory);
        }else{
            //Something happen, the creation failed I save the image inside the path.
            SaveImages.save(images.stream().map(ImageCorners::getImage).collect(Collectors.toList()), path);
            ExportProject.exportJSON(imageList, path);
        }
    }
    /*
            JSON FILE :
            {
                images = [
                            file_name : "image.tif",
                            points: [ XX:YY,
                                      XX,YY,
                                      .....
                                    ],
                              file_name : "image2.tif",
                              points: [ XX:YY,
                                        XX,YY,
                                        .....
                                       ],
                          ]
            }
         */

    /**
     * Create the json file with all the information needed
     * @param images all the images of the project
     * @return the JSONArray to write
     */
    private static JSONArray createJSON(final List<ImageCorners> images){
        final JSONArray imageList = new JSONArray();
        //Create the json
        images.forEach(imageCorners -> {
            final JSONObject obj = new JSONObject();
            final JSONArray array = new JSONArray();
            for(Point point : imageCorners.getCorners()){
                array.put(point.toString());
            }
            obj.put(POINTS_KEY, array);
            obj.put(FILE_KEY, imageCorners.getImage().getTitle());
            imageList.put(obj);
        });
        return imageList;
    }

    /**
     * Write the JSONFile in the path
     * @param array all the project information
     * @param path where will be stored the inormation
     */
    private static void exportJSON(final JSONArray array, final String path){
        try (FileWriter file = new FileWriter(path+"/"+PROJECT_NAME)) {
            file.write(array.toString());
            file.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}

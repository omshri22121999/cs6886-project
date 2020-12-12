package com.omshri.energyprofilerfordnn;

import androidx.appcompat.app.AppCompatActivity;
import androidx.constraintlayout.solver.Cache;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.webkit.URLUtil;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.AuthFailureError;
import com.android.volley.Network;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.VolleyLog;
import com.android.volley.toolbox.BasicNetwork;
import com.android.volley.toolbox.DiskBasedCache;
import com.android.volley.toolbox.HurlStack;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import com.downloader.Error;
import com.downloader.OnDownloadListener;
import com.downloader.OnProgressListener;
import com.downloader.OnStartOrResumeListener;
import com.downloader.PRDownloader;
import com.downloader.Progress;

import org.checkerframework.checker.nullness.qual.NonNull;
import org.json.JSONArray;
import org.json.JSONObject;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public class MainActivity extends AppCompatActivity {

    ProgressBar progressBar;
    EditText modellink_et;
    EditText modelname_et;
    Button download_btn;
    Button inference_btn;
    LinearLayout download_lin;
    LinearLayout inference_lin;
    TextView modelnamedisp_tv;
    ProgressBar inferenceBar;

    protected Interpreter tflite;

    private TensorImage inputImageBuffer;

    private  int imageSizeX;
    private  int imageSizeY;

    private TensorBuffer outputProbabilityBuffer;

    private static final float IMAGE_MEAN = 127.0f;
    private static final float IMAGE_STD = 128.0f;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        progressBar = findViewById(R.id.progressId);
        modellink_et = findViewById(R.id.modellink_et);
        modelname_et = findViewById(R.id.modelname_et);
        download_btn = findViewById(R.id.download_btn);
        download_lin = findViewById(R.id.download_lin);
        inference_btn = findViewById(R.id.inference_btn);
        inference_lin = findViewById(R.id.model_lin);
        modelnamedisp_tv = findViewById(R.id.modelnamedisp_tv);
        inferenceBar = findViewById(R.id.inferencesprogress);

        download_lin.setVisibility(View.INVISIBLE);
        inference_lin.setVisibility(View.GONE);
        progressBar.setIndeterminate(false);
        download_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String url = modellink_et.getText().toString();
                String name = modelname_et.getText().toString();
                if(URLUtil.isValidUrl(url)){
                    download_lin.setVisibility(View.VISIBLE);
                    String finalName = (name.equals(""))?"randommodel":name;
                    downloadModel(url,finalName);
                }
                else{
                    Toast.makeText(MainActivity.this,"Please enter valid URL",Toast.LENGTH_SHORT).show();
                }
            }
        });
        inference_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                doPreds();
            }
        });
        PRDownloader.initialize(getApplicationContext());

    }
    private void downloadModel(String url, final String modelName){
        String dirPath = getFilesDir().getAbsolutePath()+File.separator+"downloads";
        String fileName = modelName+".tflite";
        int downloadId = PRDownloader.download(url, dirPath, fileName)
                .build()
                .setOnStartOrResumeListener(new OnStartOrResumeListener() {
                    @Override
                    public void onStartOrResume() {
                        progressBar.setProgress(0);
                    }
                })
                .setOnProgressListener(new OnProgressListener() {
                    @Override
                    public void onProgress(Progress progress) {
                        progressBar.setProgress((int) (progress.currentBytes*100.0/(float)progress.totalBytes));
                    }
                })
                .start(new OnDownloadListener() {
                    @Override
                    public void onDownloadComplete() {
                        download_lin.setVisibility(View.GONE);
                        Toast.makeText(MainActivity.this,"Downloading file successful",Toast.LENGTH_SHORT).show();
                        try{
                            tflite=new Interpreter(new File(getFilesDir().getAbsolutePath()+File.separator+"downloads"+File.separator+modelName+".tflite"));
                            Toast.makeText(MainActivity.this,"Yeah done!",Toast.LENGTH_SHORT).show();
                            inference_lin.setVisibility(View.VISIBLE);
                        }catch (Exception e) {
                            e.printStackTrace();
                            Toast.makeText(MainActivity.this,e.toString(),Toast.LENGTH_SHORT).show();
                        }

                    }

                    @Override
                    public void onError(Error error) {
                        Toast.makeText(MainActivity.this,"Error while downloading!",Toast.LENGTH_SHORT).show();
                    }

                });
    }

    private TensorImage loadImage(final Bitmap bitmap) {
        // Loads bitmap into a TensorImage.
        inputImageBuffer.load(bitmap);

        // Creates processor for the TensorImage.
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        // TODO(b/143564309): Fuse ops inside ImageProcessor.
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                        .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                        .add(getPreprocessNormalizeOp())
                        .build();
        return imageProcessor.process(inputImageBuffer);
    }

    private TensorOperator getPreprocessNormalizeOp() {
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }

    public int getIndexOfLargest( float[] array )
    {
        if ( array == null || array.length == 0 ) return -1; // null or empty

        int largest = 0;
        for ( int i = 1; i < array.length; i++ )
        {
            if ( array[i] > array[largest] ) largest = i;
        }
        return largest; // position of the first largest found
    }
    private void doPreds(){

        int imageTensorIndex = 0;
        int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3}

        inferenceBar.setMax(1000);

        imageSizeY = imageShape[1];
        imageSizeX = imageShape[2];
        DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();

        int probabilityTensorIndex = 0;
        int[] probabilityShape =
                tflite.getOutputTensor(probabilityTensorIndex).shape(); // {1, NUM_CLASSES}
        DataType probabilityDataType = tflite.getOutputTensor(probabilityTensorIndex).dataType();

        inputImageBuffer = new TensorImage(imageDataType);
        outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);

        String[] images = new String[0];
        try {
            images = getAssets().list("imagen");
        } catch (IOException e) {
            e.printStackTrace();
        }
        Log.d("Images", Arrays.toString(images));
        ArrayList<Integer> arr = new ArrayList<>();
        for (String im:images) {
            Bitmap b = null;
            try {
                b = BitmapFactory.decodeStream(getAssets().open("imagen/" +im));
            } catch (IOException e) {
                e.printStackTrace();
            }
            inputImageBuffer = loadImage(b);
            tflite.run(inputImageBuffer.getBuffer(),outputProbabilityBuffer.getBuffer().rewind());
            float[] outputs = outputProbabilityBuffer.getFloatArray();
            int pred = getIndexOfLargest(outputs);
            arr.add(pred);
            Log.d("Preds", Arrays.toString(outputProbabilityBuffer.getFloatArray()));
            if(arr.size()%100==0)
                inferenceBar.setProgress(arr.size());
        }
        JSONArray j = new JSONArray(arr);
        Log.d("Predictions",j.toString());
        Toast.makeText(MainActivity.this,"Predictions Done!",Toast.LENGTH_SHORT).show();
//        sendPreds(j);
    }
    private void sendPreds(final JSONArray arr){
        RequestQueue requestQueue = Volley.newRequestQueue(MainActivity.this);
        StringRequest request = new StringRequest(Request.Method.POST, "http://192.168.0.107:5000/echo",new Response.Listener<String>() {
            @Override
            public void onResponse(String response) {
                Toast.makeText(MainActivity.this,"Sent preds!",Toast.LENGTH_SHORT).show();
            }
        }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                error.printStackTrace();
                Toast.makeText(MainActivity.this,"Error sending preds",Toast.LENGTH_SHORT).show();
            }
        }){
            @Override
            public String getBodyContentType() {
                return "application/json; charset=utf-8";
            }

            @Override
            public byte[] getBody() {
                try {
                    return arr.toString().getBytes("utf-8");
                } catch (UnsupportedEncodingException uee) {
                    VolleyLog.wtf("Unsupported Encoding while trying to get the bytes of %s using %s", arr.toString(), "utf-8");
                    return null;
                }
            }


        };

        requestQueue.add(request);
    }
}
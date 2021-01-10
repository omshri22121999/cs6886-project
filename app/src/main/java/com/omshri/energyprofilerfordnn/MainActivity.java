package com.omshri.energyprofilerfordnn;

import androidx.appcompat.app.AppCompatActivity;

import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.os.BatteryManager;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.os.Handler;
import android.util.Log;
import android.view.View;
import android.webkit.URLUtil;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.NumberPicker;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import com.downloader.Error;
import com.downloader.OnDownloadListener;
import com.downloader.OnProgressListener;
import com.downloader.OnStartOrResumeListener;
import com.downloader.PRDownloader;
import com.downloader.Progress;
import com.jaredrummler.android.device.DeviceName;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.IOException;
import java.lang.ref.WeakReference;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Timer;
import java.util.TimerTask;

public class MainActivity extends AppCompatActivity {

    ProgressBar progressBar;
    EditText modellink_et;
    EditText modelname_et;
    EditText modelmean_et;
    EditText modelstd_et;
    Button download_btn;
    Button inference_btn;
    Button idlepower_btn;
    LinearLayout download_lin;
    LinearLayout inference_lin;
    NumberPicker imageNumber;
    TextView modelnamedisp_tv;

    private BatteryManager bm;
    protected Interpreter tflite;
    private int current_readings = 0;
    private float idle_energy;
    private boolean checked_idle_energy;
    private TensorImage inputImageBuffer;
    private float file_size;

    private Boolean run_task = true;

    private  int imageSizeX;
    private  int imageSizeY;

    private TensorBuffer outputProbabilityBuffer;

    float current_average;

    private float IMAGE_MEAN = 127.0f;
    private float IMAGE_STD = 128.0f;

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        DeviceName.init(this);
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
        modelmean_et = findViewById(R.id.modelmean_et);
        imageNumber = findViewById(R.id.image_np);
        modelstd_et = findViewById(R.id.modelstd_et);
        idlepower_btn = findViewById(R.id.idlepower);

        bm = (BatteryManager) this.getSystemService(BATTERY_SERVICE);

        checked_idle_energy = false;

        imageNumber.setMaxValue(250);
        imageNumber.setMinValue(1);
        imageNumber.setValue(250);

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
                    File f = new File(getFilesDir().getAbsolutePath()+File.separator+"downloads"+File.separator+finalName+".tflite");
                    if(f.exists()){
                        file_size = f.length()/(1024.0f*1024.0f);
                        tflite=new Interpreter(f);
                        download_lin.setVisibility(View.GONE);
                        Toast.makeText(MainActivity.this,"Model exists!",Toast.LENGTH_SHORT).show();
                        inference_lin.setVisibility(View.VISIBLE);
                    }
                    else{
                        downloadModel(url,finalName);
                    }
                }
                else{
                    Toast.makeText(MainActivity.this,"Please enter valid URL",Toast.LENGTH_SHORT).show();
                }
            }
        });
        idlepower_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                current_average = 0;
                current_readings = 0;
                run_task=true;
                getCurrents();
                new CountDownTimer(20000, 1000) {
                    public void onFinish() {
                        run_task=false;
                        idle_energy = current_average*10.0f/(1000.0f*3600.0f);
                        checked_idle_energy=true;
                        Toast.makeText(getApplicationContext(),"Idle Energy : "+idle_energy,Toast.LENGTH_SHORT).show();
                        Log.d("IdleEnergy", String.valueOf(idle_energy));
                    }

                    public void onTick(long millisUntilFinished) {
                        // millisUntilFinished    The amount of time until finished.
                    }
                }.start();
            }
        });

        inference_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(checked_idle_energy) {
                    if (modelmean_et.getText().toString().equals("")) {
                        IMAGE_MEAN = 127.0f;

                    } else {
                        IMAGE_MEAN = Float.parseFloat(modelmean_et.getText().toString());
                    }
                    if (modelstd_et.getText().toString().equals("")) {
                        IMAGE_STD = 128.0f;
                    } else {
                        IMAGE_STD = Float.parseFloat(modelstd_et.getText().toString());
                    }
                    doPreds();
                }
                else{
                    Toast.makeText(getApplicationContext(),"Please check Idle Energy First",Toast.LENGTH_SHORT).show();
                }
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
                            File f = new File(getFilesDir().getAbsolutePath()+File.separator+"downloads"+File.separator+modelName+".tflite");
                            tflite=new Interpreter(f);
                            file_size = f.length()/(1024.0f*1024.0f);
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

        inference_btn.setEnabled(false);

        int imageTensorIndex = 0;
        int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3}

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
        final ArrayList<Integer> arr = new ArrayList<>();

        long startTime = Calendar.getInstance().getTimeInMillis();

        current_average = 0;
        current_readings = 0;

        run_task = true;

        getCurrents();

        for (int i=0;i<imageNumber.getValue();++i) {
            String im = images[i];
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
            Log.d("Prediction", String.valueOf(pred));
        }

        run_task=false;

        JSONArray j = new JSONArray(arr);
        Log.d("Predictions",j.toString());
        long endTime = Calendar.getInstance().getTimeInMillis();

        final Float totalTime = (float) ((endTime - startTime) / (3600.0*1000.0));

        Log.d("Time Taken",String.valueOf(totalTime));
        Log.d("Average Current",String.valueOf(current_average));
        Log.d("Charge Consumed", String.valueOf(current_average*totalTime));

        inference_btn.setEnabled(true);

        new AlertDialog.Builder(MainActivity.this)
                .setTitle("Finished Runnning Model")
                .setMessage("Battery Capacity Used = "+ current_average * totalTime / 1000.0+"mAh")

                // Specifying a listener allows you to take an action before dismissing the dialog.
                // The dialog is automatically dismissed when a dialog button is clicked.
                .setPositiveButton(android.R.string.yes, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        JSONObject details = new JSONObject();
                        try {
                            details.put("phoneName",DeviceName.getDeviceName());
                            details.put("modelLink",modellink_et.getText().toString());
                            details.put("idleEnergy",idle_energy);
                            details.put("timeTaken",totalTime*3600.0);
                            details.put("imageNumber",imageNumber.getValue());
                            details.put("energyUsed",current_average*totalTime/1000.0);
                            details.put("modelSize",file_size);
                        } catch (JSONException e) {
                            e.printStackTrace();
                        }
                        sendWhatsAppMessage(details.toString());

                    }
                })
                // A null listener allows the button to dismiss the dialog and take no further action.
                .setNegativeButton(android.R.string.no, null)
                .setIcon(android.R.drawable.ic_dialog_info)
                .show();


    }


    private void getCurrents() {
        final Timer timer = new Timer();

        timer.scheduleAtFixedRate( new TimerTask() {
            public void run() {
                if(run_task) {
                    long curr_now = bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW);
                    current_average = (float) ((current_average*current_readings + curr_now)*1.0/(current_readings+1));
                    current_readings+=1;
                }
                else{
                    timer.cancel();
                    timer.purge();
                }
            }
        }, 0, 1000);
    }

    public void sendWhatsAppMessage(String text) {

        PackageManager pm = getPackageManager();

        try {
            Log.d("Sending Message",text);
            Intent i = new Intent(Intent.ACTION_SEND);
            i.setType("text/plain");

            PackageInfo info=pm.getPackageInfo("com.whatsapp", PackageManager.GET_META_DATA);
            i.setPackage("com.whatsapp");

            i.putExtra(Intent.EXTRA_TEXT, text);
            startActivity(Intent.createChooser(i, "Share using"));

        } catch (PackageManager.NameNotFoundException e) {
            Toast.makeText(this, "WhatsApp is not installed on device.", Toast.LENGTH_SHORT)
                    .show();
        }

    }


}
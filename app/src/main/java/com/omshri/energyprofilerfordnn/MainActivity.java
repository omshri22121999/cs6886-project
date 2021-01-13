package com.omshri.energyprofilerfordnn;

import android.annotation.SuppressLint;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.os.BatteryManager;
import android.os.Bundle;
import android.os.CountDownTimer;
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

import androidx.appcompat.app.AppCompatActivity;

import com.downloader.Error;
import com.downloader.OnDownloadListener;
import com.downloader.OnProgressListener;
import com.downloader.OnStartOrResumeListener;
import com.downloader.PRDownloader;
import com.downloader.Progress;
import com.jaredrummler.android.device.DeviceName;

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
import java.util.Calendar;
import java.util.Timer;
import java.util.TimerTask;

import dmax.dialog.SpotsDialog;

public class MainActivity extends AppCompatActivity {

    ProgressBar progressBar;
    EditText modellink_et;
    EditText modelname_et;
    Button download_btn;
    Button inference_btn;
    Button idlepower_btn;
    LinearLayout download_lin;
    LinearLayout inference_lin;
    NumberPicker imageNumber;
    TextView modelnamedisp_tv;

    IntentFilter ifilter;
    Intent br;

    private BatteryManager bm;
    protected Interpreter tflite;
    private int current_readings = 0;
    private float idle_power;
    private boolean checked_idle_power;
    private TensorImage inputImageBuffer;
    private float file_size;
    private TensorBuffer outputProbabilityBuffer;

    private Boolean run_task = true;

    private int imageSizeX;
    private int imageSizeY;

    float current_average;

    private float IMAGE_MEAN = 127.5f;
    private float IMAGE_STD = 127.5f;

    String[] images;

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
        imageNumber = findViewById(R.id.image_np);
        idlepower_btn = findViewById(R.id.idlepower);

        bm = (BatteryManager) this.getSystemService(BATTERY_SERVICE);

        checked_idle_power = false;

        imageNumber.setMaxValue(250);
        imageNumber.setMinValue(1);
        imageNumber.setValue(250);

        ifilter = new IntentFilter(Intent.ACTION_BATTERY_CHANGED);
        br = this.registerReceiver(null, ifilter);

        download_lin.setVisibility(View.GONE);
        inference_lin.setVisibility(View.GONE);
        progressBar.setIndeterminate(false);
        download_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String url = modellink_et.getText().toString();
                String name = modelname_et.getText().toString();

                if (name.equals("")) {
                    Toast.makeText(MainActivity.this, "Please enter model name", Toast.LENGTH_SHORT).show();
                } else {
                    modelnamedisp_tv.setText(name.concat(".tflite"));
                    File f = new File(getFilesDir().getAbsolutePath() + File.separator + "downloads" + File.separator + name + ".tflite");
                    if (f.exists()) {
                        setupModel(name);
                        Toast.makeText(MainActivity.this, "Model exists!", Toast.LENGTH_SHORT).show();

                        inference_lin.setVisibility(View.VISIBLE);
                    } else if (URLUtil.isValidUrl(url)) {
                        download_lin.setVisibility(View.VISIBLE);
                        downloadModel(url, name);
                    } else {
                        Toast.makeText(MainActivity.this, "Please enter valid URL", Toast.LENGTH_SHORT).show();
                    }
                }
            }
        });
        idlepower_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                final AlertDialog dialog = new SpotsDialog.Builder().setContext(MainActivity.this).setCancelable(false).setMessage("Measuring Idle Power...").build();
                dialog.show();
                current_average = 0;
                current_readings = 0;
                run_task = true;
                getCurrents();
                new CountDownTimer(20000, 1000) {
                    public void onFinish() {
                        run_task = false;
                        dialog.dismiss();
                        float vol = br.getIntExtra(BatteryManager.EXTRA_VOLTAGE, -1) / 1000.0f;
                        idle_power = current_average * vol / (1000.0f);
                        checked_idle_power = true;
                        Toast.makeText(getApplicationContext(), "Completed Idle Power Calculation!", Toast.LENGTH_SHORT).show();
                        Log.d("IdlePower", String.valueOf(idle_power));
                    }

                    public void onTick(long millisUntilFinished) {

                    }
                }.start();
            }
        });

        inference_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (checked_idle_power) {
                    doPreds();
                } else {
                    Toast.makeText(getApplicationContext(), "Please check Idle Power First", Toast.LENGTH_SHORT).show();
                }
            }
        });
        PRDownloader.initialize(getApplicationContext());

    }
    private void setupModel(String modelName){

        File f = new File(getFilesDir().getAbsolutePath() + File.separator + "downloads" + File.separator + modelName + ".tflite");
        file_size = f.length() / (1024.0f * 1024.0f);
        tflite = new Interpreter(f);
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

        try {
            images = getAssets().list("imagen");
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
    private void downloadModel(String url, final String modelName) {
        String dirPath = getFilesDir().getAbsolutePath() + File.separator + "downloads";
        String fileName = modelName + ".tflite";
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
                        progressBar.setProgress((int) (progress.currentBytes * 100.0 / (float) progress.totalBytes));
                    }
                })
                .start(new OnDownloadListener() {
                    @Override
                    public void onDownloadComplete() {
                        download_lin.setVisibility(View.GONE);
                        Toast.makeText(MainActivity.this, "Downloading file successful", Toast.LENGTH_SHORT).show();
                        try {
                            setupModel(modelName);
                            inference_lin.setVisibility(View.VISIBLE);
                        } catch (Exception e) {
                            e.printStackTrace();
                            Toast.makeText(MainActivity.this, e.toString(), Toast.LENGTH_SHORT).show();
                        }

                    }

                    @Override
                    public void onError(Error error) {
                        Toast.makeText(MainActivity.this, "Error while downloading!", Toast.LENGTH_SHORT).show();
                    }

                });
    }

    private TensorImage loadImage(final Bitmap bitmap) {
        inputImageBuffer.load(bitmap);

        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());

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

    public int getIndexOfLargest(float[] array) {
        if (array == null || array.length == 0) return -1;

        int largest = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[largest]) largest = i;
        }
        return largest;
    }

    private void doPreds() {

        final AlertDialog dialog = new SpotsDialog.Builder().setContext(MainActivity.this).setCancelable(false).setMessage("Running Inferences...").build();
        dialog.show();

        final long startTime = Calendar.getInstance().getTimeInMillis();

        current_average = 0;
        current_readings = 0;

        run_task = true;

        getCurrents();

        final int vol_bef = br.getIntExtra(BatteryManager.EXTRA_VOLTAGE, -1);
        final String[] finalImages = images;
        final int num_images = imageNumber.getValue();
        new AsyncTask<Void, Void, Void>() {
            @Override
            protected Void doInBackground(Void... params) {
                for (int i = 0; i < num_images; ++i) {
                    long time_start = Calendar.getInstance().getTimeInMillis();

                    String im = finalImages[i];
                    Bitmap b = null;

                    try {
                        b = BitmapFactory.decodeStream(getAssets().open("imagen/" + im));
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    inputImageBuffer = loadImage(b);
                    tflite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());
                    float[] outputs = outputProbabilityBuffer.getFloatArray();
                }
                return null;
            }
            @Override
            protected void onPostExecute(Void aVoid) {

                super.onPostExecute(aVoid);

                dialog.dismiss();

                run_task = false;

                long endTime = Calendar.getInstance().getTimeInMillis();
                final float totalTime = (float) ((endTime - startTime) / (60*1000.0));
                final int vol_aft = br.getIntExtra(BatteryManager.EXTRA_VOLTAGE, -1);

                final float vol = (vol_bef + vol_aft) / (2000.0f);
                final float power_used = current_average * vol / 1000;

                checked_idle_power = false;

                inference_btn.setEnabled(true);
                idlepower_btn.setEnabled(true);

                new AlertDialog.Builder(MainActivity.this)
                        .setTitle("Finished Runnning Model")
                        .setMessage("Click OK to send the data via WhatsApp")
                        .setPositiveButton(android.R.string.yes, new DialogInterface.OnClickListener() {
                            @SuppressLint("StaticFieldLeak")
                            @Override
                            public void onClick(DialogInterface dialog, int which) {
                                JSONObject details = new JSONObject();
                                try {
                                    details.put("phoneName", DeviceName.getDeviceName());
                                    details.put("modelLink", modellink_et.getText().toString());
                                    details.put("idlePower", idle_power);
                                    details.put("timeTaken", totalTime);
                                    details.put("imageNumber", num_images);
                                    details.put("powerUsed", power_used);
                                    details.put("phoneVoltage", vol);
                                    details.put("modelSize", file_size);
                                    sendWhatsAppMessage(details.toString(2));
                                } catch (JSONException e) {
                                    e.printStackTrace();
                                }
                            }
                        })

                        .setNegativeButton(android.R.string.no, null)
                        .setIcon(android.R.drawable.ic_dialog_info)
                        .show();
            }
        }.execute();
    }

    private void getCurrents() {
        final Timer timer = new Timer();

        timer.scheduleAtFixedRate(new TimerTask() {
            public void run() {
                if (run_task) {
                    long curr_now = bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW);
                    current_average = ((current_average * current_readings + curr_now) * 1.0f / (current_readings + 1));
                    current_readings += 1;
                } else {
                    timer.cancel();
                    timer.purge();
                }
            }
        }, 0, 1000);
    }

    public void sendWhatsAppMessage(String text) {

        PackageManager pm = getPackageManager();

        try {
            Log.d("Sending Message", text);
            Intent i = new Intent(Intent.ACTION_SEND);
            i.setType("text/plain");

            PackageInfo info = pm.getPackageInfo("com.whatsapp", PackageManager.GET_META_DATA);
            i.setPackage("com.whatsapp");

            i.putExtra(Intent.EXTRA_TEXT, text);
            startActivity(Intent.createChooser(i, "Share using"));

        } catch (PackageManager.NameNotFoundException e) {
            Toast.makeText(this, "WhatsApp is not installed on device.", Toast.LENGTH_SHORT)
                    .show();
        }

    }
}
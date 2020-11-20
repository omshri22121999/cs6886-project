package com.omshri.energyprofilerfordnn;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.ProgressBar;
import android.widget.Toast;

import com.downloader.Error;
import com.downloader.OnCancelListener;
import com.downloader.OnDownloadListener;
import com.downloader.OnPauseListener;
import com.downloader.OnProgressListener;
import com.downloader.OnStartOrResumeListener;
import com.downloader.PRDownloader;
import com.downloader.Progress;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.URL;
import java.net.URLConnection;

public class MainActivity extends AppCompatActivity {

    ProgressBar progressBar;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        progressBar = findViewById(R.id.progressId);
//        if(downloadFile("https://tfhub.dev/tensorflow/lite-model/efficientnet/lite0/fp32/2?lite-format=tflite")){
//            Toast.makeText(this,"Downloading file successful",Toast.LENGTH_SHORT);
//        }
//        else{
//            Toast.makeText(this,"Error Downloading file",Toast.LENGTH_SHORT);
//        }
        PRDownloader.initialize(getApplicationContext());
        String url = "https://tfhub.dev/tensorflow/lite-model/efficientnet/lite0/fp32/2?lite-format=tflite";
        String dirPath = getFilesDir().getAbsolutePath()+File.separator+"downloads";
        String fileName = "model.tfite";
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
                        progressBar.setProgress((int) (progress.currentBytes/progress.totalBytes));
                    }
                })
                .start(new OnDownloadListener() {
                    @Override
                    public void onDownloadComplete() {
                        Toast.makeText(MainActivity.this,"Downloading file successful",Toast.LENGTH_SHORT).show();
                    }

                    @Override
                    public void onError(Error error) {
                        Toast.makeText(MainActivity.this,"Error while downloading!",Toast.LENGTH_SHORT).show();
                    }

                });
    }

//    public boolean downloadFile(final String path) {
//        try {
//            URL url = new URL(path);
//
//            URLConnection ucon = url.openConnection();
//            ucon.setReadTimeout(5000);
//            ucon.setConnectTimeout(10000);
//
//            InputStream is = ucon.getInputStream();
//            BufferedInputStream inStream = new BufferedInputStream(is, 1024 * 5);
//
//            File file = new File(this.getDir("filesdir", this.MODE_PRIVATE) + "/model.tflite");
//
//            if (file.exists()) {
//                file.delete();
//            }
//            file.createNewFile();
//
//            FileOutputStream outStream = new FileOutputStream(file);
//            byte[] buff = new byte[5 * 1024];
//
//            int len;
//            while ((len = inStream.read(buff)) != -1) {
//                outStream.write(buff, 0, len);
//            }
//
//            outStream.flush();
//            outStream.close();
//            inStream.close();
//
//        } catch (Exception e) {
//            e.printStackTrace();
//            return false;
//        }
//
//        return true;
//    }
}
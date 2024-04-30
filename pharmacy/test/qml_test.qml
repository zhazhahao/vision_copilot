import QtQuick 
import QtQuick.Controls 
import QtMultimedia 

ApplicationWindow {
    visible: true
    width: 640
    height: 480
    title: "Video Player"

    Video {
        id: videoPlayer
        width: parent.width
        height: parent.height
        source: "file://home/portable-00/data/video_0/20240313_160556.mp4"
        autoPlay: true
    }
}

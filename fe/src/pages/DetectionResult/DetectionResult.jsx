import React from 'react';
import ResultDisplay from '../../components/ResultDisplay/ResultDisplayNew';
import { useLocation } from 'react-router-dom';

const DetectionResult = () => {
    const location = useLocation();
    const { mode, videoFile, videoUrl } = location.state || {};

    return <ResultDisplay mode={mode} videoFile={videoFile} videoUrl={videoUrl} />;
};

export default DetectionResult;

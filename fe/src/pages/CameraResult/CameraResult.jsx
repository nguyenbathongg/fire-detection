import React from 'react';
import ResultDisplay from '../../components/ResultDisplay/ResultDisplay';
// import { useLocation } from 'react-router-dom';

const CameraResult = () => {
   // const location = useLocation();
   // const { mode } = location.state || {};

   // return <ResultDisplay mode={mode} />;
    return <ResultDisplay mode="camera" />;
};

export default CameraResult;